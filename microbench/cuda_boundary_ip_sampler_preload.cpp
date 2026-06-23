#define _GNU_SOURCE
// LD_PRELOAD sampler for instruction pointers inside cudaMemcpyAsync.
//
// Build:
//   g++ -O2 -g -fPIC -shared cuda_boundary_ip_sampler_preload.cpp \
//       -o cuda_boundary_ip_sampler_preload.so \
//       -I/usr/local/cuda/include -ldl -pthread
//
// Runtime:
//   CBIS_TRACE_FILE=/tmp/cuda_boundary_ip_samples.csv
//   CBIS_SAMPLE_NS=20000

#include <cuda_runtime_api.h>

#include <dlfcn.h>
#include <pthread.h>
#include <signal.h>
#include <string.h>
#include <sys/syscall.h>
#include <time.h>
#include <ucontext.h>
#include <unistd.h>

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <mutex>

static constexpr int kMaxSamples = 4096;

struct Sample {
    uint64_t offset_ns;
    void *ip;
};

struct Span {
    uint64_t start_ns;
    int count;
    int overflow;
    Sample samples[kMaxSamples];
};

static std::mutex g_log_mu;
static FILE *g_log = nullptr;
static std::atomic<unsigned long long> g_seq{0};
static std::once_flag g_signal_once;
static thread_local timer_t g_timer = nullptr;
static thread_local bool g_timer_ready = false;
static thread_local Span *g_active_span = nullptr;

static uint64_t now_ns()
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return uint64_t(ts.tv_sec) * 1000000000ull + uint64_t(ts.tv_nsec);
}

static int sample_signal()
{
    return SIGRTMIN + 6;
}

static long gettid_raw()
{
    return syscall(SYS_gettid);
}

static uint64_t env_u64(const char *name, uint64_t fallback)
{
    const char *value = getenv(name);
    if (!value || !*value)
        return fallback;
    return strtoull(value, nullptr, 0);
}

static FILE *log_file()
{
    std::lock_guard<std::mutex> lock(g_log_mu);
    if (g_log)
        return g_log;

    const char *path = getenv("CBIS_TRACE_FILE");
    if (!path || !*path)
        path = "/tmp/cuda_boundary_ip_samples.csv";

    g_log = fopen(path, "w");
    if (!g_log)
        return nullptr;

    fprintf(g_log,
            "record,seq,api,duration_ns,ret,count,kind,stream,sample_index,"
            "sample_offset_ns,ip,module,module_offset,symbol,symbol_offset,overflow\n");
    fflush(g_log);
    return g_log;
}

template <typename Fn>
static Fn next_symbol(const char *name)
{
    void *sym = dlsym(RTLD_NEXT, name);
    if (!sym) {
        fprintf(stderr, "cuda_boundary_ip_sampler: missing symbol %s: %s\n", name, dlerror());
        _exit(127);
    }
    return reinterpret_cast<Fn>(sym);
}

static void *sample_ip(void *ctx)
{
#if defined(__x86_64__)
    auto *uc = reinterpret_cast<ucontext_t *>(ctx);
    return reinterpret_cast<void *>(uc->uc_mcontext.gregs[REG_RIP]);
#elif defined(__aarch64__)
    auto *uc = reinterpret_cast<ucontext_t *>(ctx);
    return reinterpret_cast<void *>(uc->uc_mcontext.pc);
#else
    (void)ctx;
    return nullptr;
#endif
}

static void signal_handler(int, siginfo_t *, void *ctx)
{
    Span *span = g_active_span;
    if (!span)
        return;

    int index = span->count;
    if (index >= kMaxSamples) {
        span->overflow++;
        return;
    }

    span->samples[index].offset_ns = now_ns() - span->start_ns;
    span->samples[index].ip = sample_ip(ctx);
    span->count = index + 1;
}

static void install_signal_handler()
{
    struct sigaction sa;
    memset(&sa, 0, sizeof(sa));
    sa.sa_sigaction = signal_handler;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = SA_SIGINFO | SA_RESTART;
    if (sigaction(sample_signal(), &sa, nullptr) != 0) {
        perror("cuda_boundary_ip_sampler: sigaction");
        _exit(127);
    }
}

static void ensure_timer()
{
    std::call_once(g_signal_once, install_signal_handler);
    if (g_timer_ready)
        return;

    struct sigevent sev;
    memset(&sev, 0, sizeof(sev));
    sev.sigev_notify = SIGEV_THREAD_ID;
    sev.sigev_signo = sample_signal();
    sev._sigev_un._tid = gettid_raw();
    if (timer_create(CLOCK_MONOTONIC, &sev, &g_timer) != 0) {
        perror("cuda_boundary_ip_sampler: timer_create");
        _exit(127);
    }
    g_timer_ready = true;
}

static void arm_timer(uint64_t interval_ns)
{
    struct itimerspec its;
    memset(&its, 0, sizeof(its));
    its.it_value.tv_sec = interval_ns / 1000000000ull;
    its.it_value.tv_nsec = interval_ns % 1000000000ull;
    its.it_interval = its.it_value;
    if (timer_settime(g_timer, 0, &its, nullptr) != 0) {
        perror("cuda_boundary_ip_sampler: timer_settime(arm)");
        _exit(127);
    }
}

static void disarm_timer()
{
    struct itimerspec its;
    memset(&its, 0, sizeof(its));
    if (timer_settime(g_timer, 0, &its, nullptr) != 0) {
        perror("cuda_boundary_ip_sampler: timer_settime(disarm)");
        _exit(127);
    }
}

static void log_call_and_samples(const char *api,
                                 uint64_t duration_ns,
                                 long long ret,
                                 size_t count,
                                 int kind,
                                 cudaStream_t stream,
                                 const Span &span)
{
    FILE *f = log_file();
    if (!f)
        return;

    unsigned long long seq = ++g_seq;
    std::lock_guard<std::mutex> lock(g_log_mu);
    fprintf(f,
            "call,%llu,%s,%llu,%lld,%zu,%d,%p,,,,,,,%d\n",
            seq,
            api,
            (unsigned long long)duration_ns,
            ret,
            count,
            kind,
            stream,
            span.overflow);

    for (int i = 0; i < span.count; ++i) {
        Dl_info info;
        memset(&info, 0, sizeof(info));
        const char *module = "";
        const char *symbol = "";
        unsigned long long module_offset = 0;
        unsigned long long symbol_offset = 0;
        if (span.samples[i].ip && dladdr(span.samples[i].ip, &info)) {
            module = info.dli_fname ? info.dli_fname : "";
            symbol = info.dli_sname ? info.dli_sname : "";
            if (info.dli_fbase)
                module_offset = (unsigned long long)(
                    uintptr_t(span.samples[i].ip) - uintptr_t(info.dli_fbase));
            if (info.dli_saddr)
                symbol_offset = (unsigned long long)(
                    uintptr_t(span.samples[i].ip) - uintptr_t(info.dli_saddr));
        }

        fprintf(f,
                "sample,%llu,%s,%llu,%lld,%zu,%d,%p,%d,%llu,%p,\"%s\",%llu,\"%s\",%llu,%d\n",
                seq,
                api,
                (unsigned long long)duration_ns,
                ret,
                count,
                kind,
                stream,
                i,
                (unsigned long long)span.samples[i].offset_ns,
                span.samples[i].ip,
                module,
                module_offset,
                symbol,
                symbol_offset,
                span.overflow);
    }
    fflush(f);
}

extern "C" cudaError_t cudaMemcpyAsync(void *dst,
                                        const void *src,
                                        size_t count,
                                        enum cudaMemcpyKind kind,
                                        cudaStream_t stream)
{
    using Fn = cudaError_t (*)(void *, const void *, size_t, enum cudaMemcpyKind, cudaStream_t);
    static Fn real_fn = next_symbol<Fn>("cudaMemcpyAsync");

    ensure_timer();
    uint64_t interval_ns = env_u64("CBIS_SAMPLE_NS", 20000);
    if (interval_ns == 0)
        interval_ns = 20000;

    Span span;
    memset(&span, 0, sizeof(span));
    span.start_ns = now_ns();
    g_active_span = &span;
    arm_timer(interval_ns);
    cudaError_t ret = real_fn(dst, src, count, kind, stream);
    uint64_t end_ns = now_ns();
    disarm_timer();
    g_active_span = nullptr;

    log_call_and_samples("cudaMemcpyAsync", end_ns - span.start_ns, ret, count, int(kind), stream, span);
    return ret;
}

extern "C" cudaError_t cudaMemcpyAsync_ptsz(void *dst,
                                             const void *src,
                                             size_t count,
                                             enum cudaMemcpyKind kind,
                                             cudaStream_t stream)
{
    using Fn = cudaError_t (*)(void *, const void *, size_t, enum cudaMemcpyKind, cudaStream_t);
    static Fn real_fn = next_symbol<Fn>("cudaMemcpyAsync_ptsz");

    ensure_timer();
    uint64_t interval_ns = env_u64("CBIS_SAMPLE_NS", 20000);
    if (interval_ns == 0)
        interval_ns = 20000;

    Span span;
    memset(&span, 0, sizeof(span));
    span.start_ns = now_ns();
    g_active_span = &span;
    arm_timer(interval_ns);
    cudaError_t ret = real_fn(dst, src, count, kind, stream);
    uint64_t end_ns = now_ns();
    disarm_timer();
    g_active_span = nullptr;

    log_call_and_samples("cudaMemcpyAsync_ptsz", end_ns - span.start_ns, ret, count, int(kind), stream, span);
    return ret;
}
