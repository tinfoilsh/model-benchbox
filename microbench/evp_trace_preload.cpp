// Low-overhead LD_PRELOAD tracer for OpenSSL EVP encryption calls used by the
// NVIDIA CC launch path.
//
// The library exports evp_trace_reset() and evp_trace_dump(). The gated launch
// harness calls those hooks around the steady-state loop so the summary excludes
// CUDA initialization, graph construction, warmup, and cleanup.

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <dlfcn.h>
#include <fcntl.h>
#include <link.h>
#include <pthread.h>
#include <unistd.h>

#ifndef EVP_TRACE_ENABLE_LOADER_INTERPOSE
#define EVP_TRACE_ENABLE_LOADER_INTERPOSE 0
#endif

#ifndef EVP_TRACE_INTERPOSE_DLSYM
#define EVP_TRACE_INTERPOSE_DLSYM 0
#endif

#ifndef EVP_TRACE_ENABLE_PKCS_SLOT_PATCH
#define EVP_TRACE_ENABLE_PKCS_SLOT_PATCH 0
#endif

extern "C" {
struct evp_cipher_ctx_st;
struct evp_cipher_st;
struct engine_st;
typedef evp_cipher_ctx_st EVP_CIPHER_CTX;
typedef evp_cipher_st EVP_CIPHER;
typedef engine_st ENGINE;
typedef int CUresult;
typedef struct CUfunc_st *CUfunction;
typedef struct CUstream_st *CUstream;

const EVP_CIPHER *EVP_aes_256_gcm();
int EVP_EncryptInit_ex(EVP_CIPHER_CTX *, const EVP_CIPHER *, ENGINE *,
                       const unsigned char *, const unsigned char *);
int EVP_EncryptUpdate(EVP_CIPHER_CTX *, unsigned char *, int *,
                      const unsigned char *, int);
int EVP_EncryptFinal_ex(EVP_CIPHER_CTX *, unsigned char *, int *);
int EVP_CIPHER_CTX_ctrl(EVP_CIPHER_CTX *, int, int, void *);
#if EVP_TRACE_ENABLE_LOADER_INTERPOSE
void *dlvsym(void *handle, const char *symbol, const char *version);
#if EVP_TRACE_INTERPOSE_DLSYM
void *dlsym(void *handle, const char *symbol);
#endif
void *real_dlvsym_glibc(void *handle, const char *symbol, const char *version);
#endif
}

#if EVP_TRACE_ENABLE_LOADER_INTERPOSE
__asm__(".symver real_dlvsym_glibc, dlvsym@GLIBC_2.2.5");
#endif

namespace {

constexpr int kHistBuckets = 512;
constexpr int kSizeSlots = 128;
constexpr int kCtrlSlots = 128;

struct FnStats {
    std::atomic<unsigned long long> count{0};
    std::atomic<unsigned long long> errors{0};
    std::atomic<unsigned long long> total_ns{0};
    std::atomic<unsigned long long> max_ns{0};
    std::atomic<unsigned long long> hist_us[kHistBuckets]{};
};

struct SizeSlot {
    std::atomic<int> value{0};
    std::atomic<unsigned long long> count{0};
};

struct CtrlSlot {
    std::atomic<unsigned long long> key{0};
    std::atomic<unsigned long long> count{0};
};

struct State {
    FnStats aes_256_gcm;
    FnStats encrypt_init;
    FnStats encrypt_update;
    FnStats encrypt_final;
    FnStats cipher_ctrl;
    FnStats launch_total;
    FnStats launch_phase_a_pre_first_evp;
    FnStats launch_phase_b_evp_window;
    FnStats launch_phase_c_post_last_evp;
    FnStats launch_evp_update_sum;
    FnStats launch_evp_gap_sum;
    SizeSlot update_sizes[kSizeSlots];
    CtrlSlot ctrl_pairs[kCtrlSlots];
    std::atomic<unsigned long long> launches_without_update{0};
};

using evp_aes_256_gcm_fn = const EVP_CIPHER *(*)();
using evp_encrypt_init_ex_fn = int (*)(EVP_CIPHER_CTX *, const EVP_CIPHER *, ENGINE *,
                                      const unsigned char *, const unsigned char *);
using evp_encrypt_update_fn = int (*)(EVP_CIPHER_CTX *, unsigned char *, int *,
                                     const unsigned char *, int);
using evp_encrypt_final_ex_fn = int (*)(EVP_CIPHER_CTX *, unsigned char *, int *);
using evp_cipher_ctx_ctrl_fn = int (*)(EVP_CIPHER_CTX *, int, int, void *);
using cu_launch_kernel_fn = CUresult (*)(CUfunction, unsigned int, unsigned int,
                                        unsigned int, unsigned int, unsigned int,
                                        unsigned int, unsigned int, CUstream, void **,
                                        void **);
#if EVP_TRACE_ENABLE_LOADER_INTERPOSE
using dlsym_fn = void *(*)(void *, const char *);
#endif

State g_state;
std::atomic<bool> g_enabled{true};
std::atomic<unsigned long long> g_reset_epoch{0};
std::atomic<unsigned long long> g_dump_count{0};
thread_local bool g_in_hook = false;
thread_local bool g_in_launch_hook = false;
bool g_interpose_loader = false;
std::atomic<unsigned long long> g_patch_attempts{0};
std::atomic<unsigned long long> g_patch_successes{0};
std::atomic<unsigned long long> g_pkcs_base{0};
unsigned long long g_delay_init_ns = 0;
unsigned long long g_delay_update_ns = 0;
unsigned long long g_delay_final_ns = 0;
unsigned long long g_delay_ctrl_ns = 0;

evp_aes_256_gcm_fn real_aes_256_gcm = nullptr;
evp_encrypt_init_ex_fn real_encrypt_init = nullptr;
evp_encrypt_update_fn real_encrypt_update = nullptr;
evp_encrypt_final_ex_fn real_encrypt_final = nullptr;
evp_cipher_ctx_ctrl_fn real_cipher_ctrl = nullptr;
cu_launch_kernel_fn real_cu_launch_kernel = nullptr;
#if EVP_TRACE_ENABLE_LOADER_INTERPOSE
dlsym_fn real_dlsym = nullptr;
#endif

struct LaunchWindow {
    bool active = false;
    unsigned long long entry_ns = 0;
    unsigned long long first_update_enter_ns = 0;
    unsigned long long last_update_return_ns = 0;
    unsigned long long update_sum_ns = 0;
    unsigned long long update_count = 0;
};

thread_local LaunchWindow g_launch_window;

struct PkcsObject {
    uintptr_t base = 0;
};

int find_pkcs_object(struct dl_phdr_info *info, size_t, void *data)
{
    if (!info || !info->dlpi_name)
        return 0;
    if (!std::strstr(info->dlpi_name, "libnvidia-pkcs11-openssl3"))
        return 0;
    auto *obj = reinterpret_cast<PkcsObject *>(data);
    obj->base = static_cast<uintptr_t>(info->dlpi_addr);
    return 1;
}

bool patch_slot(uintptr_t base, uintptr_t offset, void *wrapper, void **real_slot)
{
    auto **slot = reinterpret_cast<void **>(base + offset);
    void *cur = *slot;
    if (!cur)
        return false;
    if (cur == wrapper)
        return true;
    if (!*real_slot)
        *real_slot = cur;
    *slot = wrapper;
    return true;
}

bool patch_pkcs_slots_once()
{
#if EVP_TRACE_ENABLE_PKCS_SLOT_PATCH
    g_patch_attempts.fetch_add(1, std::memory_order_relaxed);

    PkcsObject obj;
    dl_iterate_phdr(find_pkcs_object, &obj);
    if (!obj.base)
        return false;

    g_pkcs_base.store(static_cast<unsigned long long>(obj.base), std::memory_order_relaxed);

    bool ok = true;
    ok &= patch_slot(obj.base, 0x4090, reinterpret_cast<void *>(&EVP_aes_256_gcm),
                     reinterpret_cast<void **>(&real_aes_256_gcm));
    ok &= patch_slot(obj.base, 0x40d0, reinterpret_cast<void *>(&EVP_EncryptInit_ex),
                     reinterpret_cast<void **>(&real_encrypt_init));
    ok &= patch_slot(obj.base, 0x40c8, reinterpret_cast<void *>(&EVP_EncryptUpdate),
                     reinterpret_cast<void **>(&real_encrypt_update));
    ok &= patch_slot(obj.base, 0x40c0, reinterpret_cast<void *>(&EVP_EncryptFinal_ex),
                     reinterpret_cast<void **>(&real_encrypt_final));
    ok &= patch_slot(obj.base, 0x40a0, reinterpret_cast<void *>(&EVP_CIPHER_CTX_ctrl),
                     reinterpret_cast<void **>(&real_cipher_ctrl));
    if (ok)
        g_patch_successes.fetch_add(1, std::memory_order_relaxed);
    return ok;
#else
    return false;
#endif
}

void *patch_thread_main(void *)
{
#if EVP_TRACE_ENABLE_PKCS_SLOT_PATCH
    const char *patch_env = std::getenv("EVP_TRACE_PATCH_PKCS_SLOTS");
    if (!patch_env || std::strcmp(patch_env, "0") == 0)
        return nullptr;

    for (int i = 0; i < 2000; ++i) {
        if (patch_pkcs_slots_once())
            return nullptr;
        usleep(5000);
    }
#endif
    return nullptr;
}

unsigned long long now_ns()
{
    timespec ts{};
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
    return static_cast<unsigned long long>(ts.tv_sec) * 1000000000ull +
           static_cast<unsigned long long>(ts.tv_nsec);
}

unsigned long long env_ull(const char *name, unsigned long long default_value)
{
    const char *value = std::getenv(name);
    if (!value || !*value)
        return default_value;
    char *end = nullptr;
    unsigned long long parsed = std::strtoull(value, &end, 0);
    return end && *end == '\0' ? parsed : default_value;
}

void busy_wait_ns(unsigned long long delay_ns)
{
    if (!delay_ns)
        return;
    unsigned long long target = now_ns() + delay_ns;
    while (now_ns() < target)
        asm volatile("pause" ::: "memory");
}

void *raw_dlsym(void *handle, const char *symbol)
{
#if EVP_TRACE_ENABLE_LOADER_INTERPOSE
    if (!real_dlsym)
        real_dlsym = reinterpret_cast<dlsym_fn>(real_dlvsym_glibc(RTLD_NEXT, "dlsym", "GLIBC_2.2.5"));
    return real_dlsym ? real_dlsym(handle, symbol) : nullptr;
#else
    return dlsym(handle, symbol);
#endif
}

#if EVP_TRACE_ENABLE_LOADER_INTERPOSE
void *raw_dlvsym(void *handle, const char *symbol, const char *version)
{
    return real_dlvsym_glibc(handle, symbol, version);
}

void *maybe_interpose_crypto_symbol(void *real, const char *symbol)
{
    if (!symbol)
        return nullptr;
    if (std::strcmp(symbol, "EVP_aes_256_gcm") == 0) {
        if (real)
            real_aes_256_gcm = reinterpret_cast<evp_aes_256_gcm_fn>(real);
        return real_aes_256_gcm ? reinterpret_cast<void *>(&EVP_aes_256_gcm) : nullptr;
    }
    if (std::strcmp(symbol, "EVP_EncryptInit_ex") == 0) {
        if (real)
            real_encrypt_init = reinterpret_cast<evp_encrypt_init_ex_fn>(real);
        return real_encrypt_init ? reinterpret_cast<void *>(&EVP_EncryptInit_ex) : nullptr;
    }
    if (std::strcmp(symbol, "EVP_EncryptUpdate") == 0) {
        if (real)
            real_encrypt_update = reinterpret_cast<evp_encrypt_update_fn>(real);
        return real_encrypt_update ? reinterpret_cast<void *>(&EVP_EncryptUpdate) : nullptr;
    }
    if (std::strcmp(symbol, "EVP_EncryptFinal_ex") == 0) {
        if (real)
            real_encrypt_final = reinterpret_cast<evp_encrypt_final_ex_fn>(real);
        return real_encrypt_final ? reinterpret_cast<void *>(&EVP_EncryptFinal_ex) : nullptr;
    }
    if (std::strcmp(symbol, "EVP_CIPHER_CTX_ctrl") == 0) {
        if (real)
            real_cipher_ctrl = reinterpret_cast<evp_cipher_ctx_ctrl_fn>(real);
        return real_cipher_ctrl ? reinterpret_cast<void *>(&EVP_CIPHER_CTX_ctrl) : nullptr;
    }
    return nullptr;
}
#endif

void resolve_symbols()
{
    if (!real_aes_256_gcm)
        real_aes_256_gcm = reinterpret_cast<evp_aes_256_gcm_fn>(raw_dlsym(RTLD_NEXT, "EVP_aes_256_gcm"));
    if (!real_encrypt_init)
        real_encrypt_init = reinterpret_cast<evp_encrypt_init_ex_fn>(raw_dlsym(RTLD_NEXT, "EVP_EncryptInit_ex"));
    if (!real_encrypt_update)
        real_encrypt_update = reinterpret_cast<evp_encrypt_update_fn>(raw_dlsym(RTLD_NEXT, "EVP_EncryptUpdate"));
    if (!real_encrypt_final)
        real_encrypt_final = reinterpret_cast<evp_encrypt_final_ex_fn>(raw_dlsym(RTLD_NEXT, "EVP_EncryptFinal_ex"));
    if (!real_cipher_ctrl)
        real_cipher_ctrl = reinterpret_cast<evp_cipher_ctx_ctrl_fn>(raw_dlsym(RTLD_NEXT, "EVP_CIPHER_CTX_ctrl"));
}

void resolve_launch_symbol()
{
    if (!real_cu_launch_kernel)
        real_cu_launch_kernel = reinterpret_cast<cu_launch_kernel_fn>(raw_dlsym(RTLD_NEXT, "cuLaunchKernel"));
    if (!real_cu_launch_kernel) {
        void *cuda = dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
        if (cuda)
            real_cu_launch_kernel = reinterpret_cast<cu_launch_kernel_fn>(raw_dlsym(cuda, "cuLaunchKernel"));
    }
}

void update_max(std::atomic<unsigned long long> &slot, unsigned long long value)
{
    unsigned long long old = slot.load(std::memory_order_relaxed);
    while (old < value && !slot.compare_exchange_weak(old, value, std::memory_order_relaxed)) {
    }
}

void record_value_ns(FnStats &stats, unsigned long long dt, int ok)
{
    stats.count.fetch_add(1, std::memory_order_relaxed);
    if (!ok)
        stats.errors.fetch_add(1, std::memory_order_relaxed);
    stats.total_ns.fetch_add(dt, std::memory_order_relaxed);
    update_max(stats.max_ns, dt);
    unsigned long long bucket = dt / 1000;
    if (bucket >= kHistBuckets)
        bucket = kHistBuckets - 1;
    stats.hist_us[bucket].fetch_add(1, std::memory_order_relaxed);
}

void record_duration(FnStats &stats, unsigned long long start_ns, int ok)
{
    record_value_ns(stats, now_ns() - start_ns, ok);
}

void record_size(int value)
{
    for (int i = 0; i < kSizeSlots; ++i) {
        int cur = g_state.update_sizes[i].value.load(std::memory_order_relaxed);
        if (cur == value) {
            g_state.update_sizes[i].count.fetch_add(1, std::memory_order_relaxed);
            return;
        }
        if (cur == 0 && g_state.update_sizes[i].value.compare_exchange_strong(cur, value, std::memory_order_relaxed)) {
            g_state.update_sizes[i].count.store(1, std::memory_order_relaxed);
            return;
        }
    }
}

void record_ctrl_pair(int type, int arg)
{
    unsigned long long key =
        (static_cast<unsigned long long>(static_cast<unsigned int>(type)) << 32) |
        static_cast<unsigned int>(arg);
    for (int i = 0; i < kCtrlSlots; ++i) {
        unsigned long long cur = g_state.ctrl_pairs[i].key.load(std::memory_order_relaxed);
        if (cur == key) {
            g_state.ctrl_pairs[i].count.fetch_add(1, std::memory_order_relaxed);
            return;
        }
        if (cur == 0 && g_state.ctrl_pairs[i].key.compare_exchange_strong(cur, key, std::memory_order_relaxed)) {
            g_state.ctrl_pairs[i].count.store(1, std::memory_order_relaxed);
            return;
        }
    }
}

unsigned long long percentile_us(const FnStats &stats, double q)
{
    unsigned long long n = stats.count.load(std::memory_order_relaxed);
    if (n == 0)
        return 0;
    unsigned long long want = static_cast<unsigned long long>(q * static_cast<double>(n - 1)) + 1;
    unsigned long long seen = 0;
    for (int i = 0; i < kHistBuckets; ++i) {
        seen += stats.hist_us[i].load(std::memory_order_relaxed);
        if (seen >= want)
            return static_cast<unsigned long long>(i);
    }
    return kHistBuckets - 1;
}

void reset_fn(FnStats &stats)
{
    stats.count.store(0, std::memory_order_relaxed);
    stats.errors.store(0, std::memory_order_relaxed);
    stats.total_ns.store(0, std::memory_order_relaxed);
    stats.max_ns.store(0, std::memory_order_relaxed);
    for (int i = 0; i < kHistBuckets; ++i)
        stats.hist_us[i].store(0, std::memory_order_relaxed);
}

void reset_state()
{
    g_enabled.store(false, std::memory_order_seq_cst);
    reset_fn(g_state.aes_256_gcm);
    reset_fn(g_state.encrypt_init);
    reset_fn(g_state.encrypt_update);
    reset_fn(g_state.encrypt_final);
    reset_fn(g_state.cipher_ctrl);
    reset_fn(g_state.launch_total);
    reset_fn(g_state.launch_phase_a_pre_first_evp);
    reset_fn(g_state.launch_phase_b_evp_window);
    reset_fn(g_state.launch_phase_c_post_last_evp);
    reset_fn(g_state.launch_evp_update_sum);
    reset_fn(g_state.launch_evp_gap_sum);
    g_state.launches_without_update.store(0, std::memory_order_relaxed);
    for (int i = 0; i < kSizeSlots; ++i) {
        g_state.update_sizes[i].value.store(0, std::memory_order_relaxed);
        g_state.update_sizes[i].count.store(0, std::memory_order_relaxed);
    }
    for (int i = 0; i < kCtrlSlots; ++i) {
        g_state.ctrl_pairs[i].key.store(0, std::memory_order_relaxed);
        g_state.ctrl_pairs[i].count.store(0, std::memory_order_relaxed);
    }
    g_reset_epoch.fetch_add(1, std::memory_order_relaxed);
    g_enabled.store(true, std::memory_order_seq_cst);
}

FILE *open_output()
{
    const char *path = std::getenv("EVP_TRACE_OUT");
    if (!path || !*path)
        return stderr;
    FILE *f = std::fopen(path, "a");
    return f ? f : stderr;
}

void print_fn(FILE *f, const char *name, const FnStats &stats)
{
    unsigned long long count = stats.count.load(std::memory_order_relaxed);
    unsigned long long total = stats.total_ns.load(std::memory_order_relaxed);
    double mean_ns = count ? static_cast<double>(total) / static_cast<double>(count) : 0.0;
    std::fprintf(f,
                 "function\t%s\tcount=%llu\terrors=%llu\ttotal_ns=%llu\tmean_ns=%.1f\tp50_us=%llu\tp90_us=%llu\tp99_us=%llu\tmax_ns=%llu\n",
                 name,
                 count,
                 stats.errors.load(std::memory_order_relaxed),
                 total,
                 mean_ns,
                 percentile_us(stats, 0.50),
                 percentile_us(stats, 0.90),
                 percentile_us(stats, 0.99),
                 stats.max_ns.load(std::memory_order_relaxed));
}

void dump_state()
{
    g_enabled.store(false, std::memory_order_seq_cst);
    FILE *f = open_output();
    unsigned long long dump_id = g_dump_count.fetch_add(1, std::memory_order_relaxed) + 1;
    const char *label = std::getenv("EVP_TRACE_LABEL");
    if (!label)
        label = "";

    std::fprintf(f,
                 "# evp_trace\tpid=%d\tdump=%llu\treset_epoch=%llu\tlabel=%s\ttime_ns=%llu\n",
                 getpid(),
                 dump_id,
                 g_reset_epoch.load(std::memory_order_relaxed),
                 label,
                 now_ns());
    std::fprintf(f,
                 "patch\tattempts=%llu\tsuccesses=%llu\tpkcs_base=0x%llx\n",
                 g_patch_attempts.load(std::memory_order_relaxed),
                 g_patch_successes.load(std::memory_order_relaxed),
                 g_pkcs_base.load(std::memory_order_relaxed));
    std::fprintf(f,
                 "delay_ns\tinit=%llu\tupdate=%llu\tfinal=%llu\tctrl=%llu\n",
                 g_delay_init_ns,
                 g_delay_update_ns,
                 g_delay_final_ns,
                 g_delay_ctrl_ns);
    print_fn(f, "EVP_aes_256_gcm", g_state.aes_256_gcm);
    print_fn(f, "EVP_EncryptInit_ex", g_state.encrypt_init);
    print_fn(f, "EVP_EncryptUpdate", g_state.encrypt_update);
    print_fn(f, "EVP_EncryptFinal_ex", g_state.encrypt_final);
    print_fn(f, "EVP_CIPHER_CTX_ctrl", g_state.cipher_ctrl);
    print_fn(f, "cuLaunchKernel_total", g_state.launch_total);
    print_fn(f, "cuLaunchKernel_A_entry_to_first_evp", g_state.launch_phase_a_pre_first_evp);
    print_fn(f, "cuLaunchKernel_B_first_evp_to_last_evp_return", g_state.launch_phase_b_evp_window);
    print_fn(f, "cuLaunchKernel_C_last_evp_return_to_return", g_state.launch_phase_c_post_last_evp);
    print_fn(f, "cuLaunchKernel_evp_update_sum", g_state.launch_evp_update_sum);
    print_fn(f, "cuLaunchKernel_B_minus_evp_update_sum", g_state.launch_evp_gap_sum);
    std::fprintf(f,
                 "launch_phase\tlaunches_without_evp_update=%llu\n",
                 g_state.launches_without_update.load(std::memory_order_relaxed));

    for (int i = 0; i < kSizeSlots; ++i) {
        int value = g_state.update_sizes[i].value.load(std::memory_order_relaxed);
        unsigned long long count = g_state.update_sizes[i].count.load(std::memory_order_relaxed);
        if (value != 0 && count != 0)
            std::fprintf(f, "update_size\t0x%x\t%d\t%llu\n", value, value, count);
    }
    for (int i = 0; i < kCtrlSlots; ++i) {
        unsigned long long key = g_state.ctrl_pairs[i].key.load(std::memory_order_relaxed);
        unsigned long long count = g_state.ctrl_pairs[i].count.load(std::memory_order_relaxed);
        if (key != 0 && count != 0) {
            unsigned int type = static_cast<unsigned int>(key >> 32);
            unsigned int arg = static_cast<unsigned int>(key & 0xffffffffu);
            std::fprintf(f, "ctrl_pair\t0x%x\t0x%x\t%llu\n", type, arg, count);
        }
    }
    std::fprintf(f, "# evp_trace_end\n");
    if (f != stderr)
        std::fclose(f);
    else
        std::fflush(f);
    g_enabled.store(true, std::memory_order_seq_cst);
}

} // namespace

extern "C" void evp_trace_reset()
{
    reset_state();
}

extern "C" void evp_trace_dump()
{
    dump_state();
}

#if EVP_TRACE_ENABLE_LOADER_INTERPOSE
#if EVP_TRACE_INTERPOSE_DLSYM
extern "C" void *dlsym(void *handle, const char *symbol)
{
    void *real = raw_dlsym(handle, symbol);
    if (g_interpose_loader) {
        if (void *wrapped = maybe_interpose_crypto_symbol(real, symbol))
            return wrapped;
    }
    return real;
}
#endif

extern "C" void *dlvsym(void *handle, const char *symbol, const char *version)
{
    void *real = raw_dlvsym(handle, symbol, version);
    if (g_interpose_loader) {
        if (void *wrapped = maybe_interpose_crypto_symbol(real, symbol))
            return wrapped;
    }
    return real;
}
#endif

extern "C" const EVP_CIPHER *EVP_aes_256_gcm()
{
    resolve_symbols();
    if (!real_aes_256_gcm)
        return nullptr;
    if (g_in_hook || !g_enabled.load(std::memory_order_relaxed))
        return real_aes_256_gcm();
    g_in_hook = true;
    unsigned long long t0 = now_ns();
    const EVP_CIPHER *ret = real_aes_256_gcm();
    record_duration(g_state.aes_256_gcm, t0, ret != nullptr);
    g_in_hook = false;
    return ret;
}

extern "C" int EVP_EncryptInit_ex(EVP_CIPHER_CTX *ctx, const EVP_CIPHER *type,
                                  ENGINE *impl, const unsigned char *key,
                                  const unsigned char *iv)
{
    resolve_symbols();
    if (!real_encrypt_init)
        return 0;
    if (g_in_hook || !g_enabled.load(std::memory_order_relaxed))
        return real_encrypt_init(ctx, type, impl, key, iv);
    g_in_hook = true;
    unsigned long long t0 = now_ns();
    int ret = real_encrypt_init(ctx, type, impl, key, iv);
    busy_wait_ns(g_delay_init_ns);
    record_duration(g_state.encrypt_init, t0, ret == 1);
    g_in_hook = false;
    return ret;
}

extern "C" int EVP_EncryptUpdate(EVP_CIPHER_CTX *ctx, unsigned char *out, int *outl,
                                 const unsigned char *in, int inl)
{
    resolve_symbols();
    if (!real_encrypt_update)
        return 0;
    if (g_in_hook || !g_enabled.load(std::memory_order_relaxed))
        return real_encrypt_update(ctx, out, outl, in, inl);
    g_in_hook = true;
    unsigned long long t0 = now_ns();
    if (g_launch_window.active && g_launch_window.update_count == 0)
        g_launch_window.first_update_enter_ns = t0;
    int ret = real_encrypt_update(ctx, out, outl, in, inl);
    busy_wait_ns(g_delay_update_ns);
    unsigned long long t1 = now_ns();
    if (g_launch_window.active) {
        g_launch_window.last_update_return_ns = t1;
        g_launch_window.update_sum_ns += t1 - t0;
        ++g_launch_window.update_count;
    }
    record_size(inl);
    record_value_ns(g_state.encrypt_update, t1 - t0, ret == 1);
    g_in_hook = false;
    return ret;
}

extern "C" int EVP_EncryptFinal_ex(EVP_CIPHER_CTX *ctx, unsigned char *out, int *outl)
{
    resolve_symbols();
    if (!real_encrypt_final)
        return 0;
    if (g_in_hook || !g_enabled.load(std::memory_order_relaxed))
        return real_encrypt_final(ctx, out, outl);
    g_in_hook = true;
    unsigned long long t0 = now_ns();
    int ret = real_encrypt_final(ctx, out, outl);
    busy_wait_ns(g_delay_final_ns);
    record_duration(g_state.encrypt_final, t0, ret == 1);
    g_in_hook = false;
    return ret;
}

extern "C" int EVP_CIPHER_CTX_ctrl(EVP_CIPHER_CTX *ctx, int type, int arg, void *ptr)
{
    resolve_symbols();
    if (!real_cipher_ctrl)
        return 0;
    if (g_in_hook || !g_enabled.load(std::memory_order_relaxed))
        return real_cipher_ctrl(ctx, type, arg, ptr);
    g_in_hook = true;
    unsigned long long t0 = now_ns();
    int ret = real_cipher_ctrl(ctx, type, arg, ptr);
    busy_wait_ns(g_delay_ctrl_ns);
    record_ctrl_pair(type, arg);
    record_duration(g_state.cipher_ctrl, t0, ret == 1);
    g_in_hook = false;
    return ret;
}

extern "C" CUresult cuLaunchKernel(CUfunction f,
                                    unsigned int gridDimX,
                                    unsigned int gridDimY,
                                    unsigned int gridDimZ,
                                    unsigned int blockDimX,
                                    unsigned int blockDimY,
                                    unsigned int blockDimZ,
                                    unsigned int sharedMemBytes,
                                    CUstream hStream,
                                    void **kernelParams,
                                    void **extra)
{
    resolve_launch_symbol();
    if (!real_cu_launch_kernel) {
        std::fprintf(stderr, "evp_trace_preload: missing real cuLaunchKernel\n");
        _exit(127);
    }

    if (g_in_launch_hook || !g_enabled.load(std::memory_order_relaxed)) {
        return real_cu_launch_kernel(f,
                                     gridDimX,
                                     gridDimY,
                                     gridDimZ,
                                     blockDimX,
                                     blockDimY,
                                     blockDimZ,
                                     sharedMemBytes,
                                     hStream,
                                     kernelParams,
                                     extra);
    }

    g_in_launch_hook = true;
    LaunchWindow previous = g_launch_window;
    g_launch_window = LaunchWindow{};
    g_launch_window.active = true;
    g_launch_window.entry_ns = now_ns();

    CUresult ret = real_cu_launch_kernel(f,
                                         gridDimX,
                                         gridDimY,
                                         gridDimZ,
                                         blockDimX,
                                         blockDimY,
                                         blockDimZ,
                                         sharedMemBytes,
                                         hStream,
                                         kernelParams,
                                         extra);
    unsigned long long return_ns = now_ns();

    LaunchWindow window = g_launch_window;
    g_launch_window = previous;
    g_in_launch_hook = false;

    unsigned long long total_ns = return_ns - window.entry_ns;
    record_value_ns(g_state.launch_total, total_ns, ret == 0);

    if (window.update_count > 0 && window.first_update_enter_ns >= window.entry_ns &&
        window.last_update_return_ns >= window.first_update_enter_ns &&
        return_ns >= window.last_update_return_ns) {
        unsigned long long a_ns = window.first_update_enter_ns - window.entry_ns;
        unsigned long long b_ns = window.last_update_return_ns - window.first_update_enter_ns;
        unsigned long long c_ns = return_ns - window.last_update_return_ns;
        unsigned long long gap_ns = b_ns > window.update_sum_ns ? b_ns - window.update_sum_ns : 0;

        record_value_ns(g_state.launch_phase_a_pre_first_evp, a_ns, true);
        record_value_ns(g_state.launch_phase_b_evp_window, b_ns, true);
        record_value_ns(g_state.launch_phase_c_post_last_evp, c_ns, true);
        record_value_ns(g_state.launch_evp_update_sum, window.update_sum_ns, true);
        record_value_ns(g_state.launch_evp_gap_sum, gap_ns, true);
    } else {
        g_state.launches_without_update.fetch_add(1, std::memory_order_relaxed);
    }

    return ret;
}

__attribute__((constructor)) static void evp_trace_ctor()
{
    g_delay_init_ns = env_ull("EVP_TRACE_DELAY_INIT_NS", 0);
    g_delay_update_ns = env_ull("EVP_TRACE_DELAY_UPDATE_NS", 0);
    g_delay_final_ns = env_ull("EVP_TRACE_DELAY_FINAL_NS", 0);
    g_delay_ctrl_ns = env_ull("EVP_TRACE_DELAY_CTRL_NS", 0);

    const char *start_disabled = std::getenv("EVP_TRACE_START_DISABLED");
    if (start_disabled && std::strcmp(start_disabled, "0") != 0)
        g_enabled.store(false, std::memory_order_relaxed);

#if EVP_TRACE_ENABLE_LOADER_INTERPOSE
    const char *interpose_loader = std::getenv("EVP_TRACE_INTERPOSE_LOADER");
    g_interpose_loader = interpose_loader && std::strcmp(interpose_loader, "0") != 0;
#endif

#if EVP_TRACE_ENABLE_PKCS_SLOT_PATCH
    pthread_t thread;
    if (pthread_create(&thread, nullptr, patch_thread_main, nullptr) == 0)
        pthread_detach(thread);
#endif
}

__attribute__((destructor)) static void evp_trace_dtor()
{
    const char *autodump = std::getenv("EVP_TRACE_AUTODUMP");
    if (autodump && std::strcmp(autodump, "0") != 0)
        dump_state();
}
