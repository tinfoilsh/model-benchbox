// Probe whether GPU writes to mapped pinned host memory avoid CC D2H boundaries.
//
// Build:
//   g++ -O2 -g cuda_mapped_host_mailbox_probe.cpp \
//       -o cuda_mapped_host_mailbox_probe \
//       -I/usr/local/cuda/include -L/usr/local/cuda/lib64 \
//       -lcudart -lcuda -Wl,-rpath,/usr/local/cuda/lib64
//
// Runtime env:
//   CMM_PAYLOAD_BYTES=4,256,4096,65536
//   CMM_KERNEL_CYCLES=2000000
//   CMM_REPEATS=5
//   CMM_WARMUP=1
//   CMM_OUT=/out/cuda_mapped_host_mailbox.csv

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#define CUDA_CHECK(expr)                                                       \
    do {                                                                       \
        cudaError_t _err = (expr);                                             \
        if (_err != cudaSuccess) {                                             \
            fprintf(stderr,                                                    \
                    "CUDA error at %s:%d: %s returned %s (%d)\n",             \
                    __FILE__,                                                  \
                    __LINE__,                                                  \
                    #expr,                                                     \
                    cudaGetErrorString(_err),                                  \
                    int(_err));                                                \
            return 1;                                                          \
        }                                                                      \
    } while (0)

#define CU_CHECK(expr)                                                         \
    do {                                                                       \
        CUresult _err = (expr);                                                \
        if (_err != CUDA_SUCCESS) {                                            \
            const char *name = nullptr;                                        \
            const char *text = nullptr;                                        \
            cuGetErrorName(_err, &name);                                       \
            cuGetErrorString(_err, &text);                                     \
            fprintf(stderr,                                                    \
                    "CU error at %s:%d: %s returned %s: %s (%d)\n",           \
                    __FILE__,                                                  \
                    __LINE__,                                                  \
                    #expr,                                                     \
                    name ? name : "unknown",                                  \
                    text ? text : "unknown",                                  \
                    int(_err));                                                \
            return 1;                                                          \
        }                                                                      \
    } while (0)

static const char *kPtx = R"PTX(
.version 8.0
.target sm_90
.address_size 64

.visible .entry mapped_mailbox_kernel(
    .param .u64 host_ptr,
    .param .u32 payload_words,
    .param .u64 cycles,
    .param .u32 value
)
{
    .reg .pred %p;
    .reg .u64 %host, %cycles, %start, %now, %delta, %addr, %idx64, %off;
    .reg .u32 %words, %value, %i;

    ld.param.u64 %host, [host_ptr];
    ld.param.u32 %words, [payload_words];
    ld.param.u64 %cycles, [cycles];
    ld.param.u32 %value, [value];

    mov.u64 %start, %clock64;
spin:
    mov.u64 %now, %clock64;
    sub.u64 %delta, %now, %start;
    setp.lt.u64 %p, %delta, %cycles;
    @%p bra spin;

    mov.u32 %i, 1;
write_loop:
    setp.ge.u32 %p, %i, %words;
    @%p bra write_done;
    cvt.u64.u32 %idx64, %i;
    mul.lo.u64 %off, %idx64, 4;
    add.u64 %addr, %host, %off;
    st.global.u32 [%addr], %value;
    add.u32 %i, %i, 1;
    bra write_loop;

write_done:
    membar.sys;
    st.global.u32 [%host], %value;
    membar.sys;
    ret;
}

.visible .entry spin_kernel(
    .param .u64 cycles
)
{
    .reg .pred %p;
    .reg .u64 %cycles, %start, %now, %delta;
    ld.param.u64 %cycles, [cycles];
    mov.u64 %start, %clock64;
spin2:
    mov.u64 %now, %clock64;
    sub.u64 %delta, %now, %start;
    setp.lt.u64 %p, %delta, %cycles;
    @%p bra spin2;
    ret;
}
)PTX";

static unsigned long long env_ull(const char *name, unsigned long long fallback)
{
    const char *value = getenv(name);
    if (!value || !*value)
        return fallback;
    return strtoull(value, nullptr, 0);
}

static std::vector<size_t> env_size_list(const char *name, const char *fallback)
{
    const char *value = getenv(name);
    std::string text = (value && *value) ? value : fallback;
    std::vector<size_t> out;
    std::stringstream ss(text);
    std::string item;
    while (std::getline(ss, item, ',')) {
        if (!item.empty())
            out.push_back(size_t(strtoull(item.c_str(), nullptr, 0)));
    }
    return out;
}

static double us_since(std::chrono::steady_clock::time_point end,
                       std::chrono::steady_clock::time_point start)
{
    return std::chrono::duration<double, std::micro>(end - start).count();
}

static int launch_mailbox(CUfunction fn,
                          cudaStream_t stream,
                          void *host_dev_ptr,
                          unsigned int payload_words,
                          unsigned long long cycles,
                          unsigned int value)
{
    void *args[] = {&host_dev_ptr, &payload_words, &cycles, &value};
    CU_CHECK(cuLaunchKernel(fn,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                            0,
                            reinterpret_cast<CUstream>(stream),
                            args,
                            nullptr));
    return 0;
}

static int launch_spin(CUfunction fn, cudaStream_t stream, unsigned long long cycles)
{
    void *args[] = {&cycles};
    CU_CHECK(cuLaunchKernel(fn,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                            0,
                            reinterpret_cast<CUstream>(stream),
                            args,
                            nullptr));
    return 0;
}

static void write_header(FILE *out)
{
    fprintf(out,
            "scenario,payload_bytes,iteration,launch_api_us,wait_api_us,"
            "post_sync_us,total_us,poll_count,observed_value\n");
}

int main()
{
    const std::vector<size_t> payload_sizes = env_size_list("CMM_PAYLOAD_BYTES", "4,256,4096,65536");
    if (payload_sizes.empty()) {
        fprintf(stderr, "no payload sizes\n");
        return 1;
    }
    const size_t max_payload = *std::max_element(payload_sizes.begin(), payload_sizes.end());
    const unsigned long long cycles = env_ull("CMM_KERNEL_CYCLES", 2000000ull);
    const int repeats = int(env_ull("CMM_REPEATS", 5ull));
    const int warmup = int(env_ull("CMM_WARMUP", 1ull));
    const int total_iters = warmup + repeats;
    const char *out_path = getenv("CMM_OUT");
    if (!out_path || !*out_path)
        out_path = "/tmp/cuda_mapped_host_mailbox.csv";

    CUDA_CHECK(cudaSetDeviceFlags(cudaDeviceMapHost));
    CUDA_CHECK(cudaSetDevice(0));
    CU_CHECK(cuInit(0));

    CUmodule module = nullptr;
    CUfunction mailbox_fn = nullptr;
    CUfunction spin_fn = nullptr;
    CU_CHECK(cuModuleLoadData(&module, kPtx));
    CU_CHECK(cuModuleGetFunction(&mailbox_fn, module, "mapped_mailbox_kernel"));
    CU_CHECK(cuModuleGetFunction(&spin_fn, module, "spin_kernel"));

    void *host = nullptr;
    void *host_dev = nullptr;
    const size_t host_bytes = std::max<size_t>(4096, max_payload);
    CUDA_CHECK(cudaHostAlloc(&host, host_bytes, cudaHostAllocMapped));
    CUDA_CHECK(cudaHostGetDevicePointer(&host_dev, host, 0));
    memset(host, 0, host_bytes);

    void *dev_token = nullptr;
    void *normal_host = nullptr;
    CUDA_CHECK(cudaMalloc(&dev_token, 4096));
    CUDA_CHECK(cudaHostAlloc(&normal_host, 4096, cudaHostAllocDefault));

    cudaStream_t stream = nullptr;
    cudaEvent_t event = nullptr;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
    CUDA_CHECK(cudaMemsetAsync(dev_token, 0x5A, 4096, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    FILE *out = fopen(out_path, "w");
    if (!out) {
        perror("fopen(CMM_OUT)");
        return 1;
    }
    write_header(out);

    auto clear_host = [&]() {
        memset(host, 0, host_bytes);
        __sync_synchronize();
    };

    auto emit = [&](const char *scenario,
                    size_t payload_bytes,
                    int iteration,
                    double launch_api_us,
                    double wait_api_us,
                    double post_sync_us,
                    double total_us,
                    unsigned long long poll_count,
                    unsigned int observed) {
        fprintf(out,
                "%s,%zu,%d,%.3f,%.3f,%.3f,%.3f,%llu,%u\n",
                scenario,
                payload_bytes,
                iteration,
                launch_api_us,
                wait_api_us,
                post_sync_us,
                total_us,
                poll_count,
                observed);
    };

    for (size_t payload_bytes : payload_sizes) {
        const unsigned int payload_words =
            std::max<unsigned int>(1, static_cast<unsigned int>((payload_bytes + 3) / 4));

        for (int i = 0; i < total_iters; ++i) {
            clear_host();
            CUDA_CHECK(cudaStreamSynchronize(stream));
            const unsigned int value = 0x10000000u + unsigned(i);
            auto t0 = std::chrono::steady_clock::now();
            if (launch_mailbox(mailbox_fn, stream, host_dev, payload_words, cycles, value) != 0)
                return 1;
            auto t1 = std::chrono::steady_clock::now();
            CUDA_CHECK(cudaStreamSynchronize(stream));
            auto t2 = std::chrono::steady_clock::now();
            unsigned int observed = *reinterpret_cast<volatile unsigned int *>(host);
            if (i >= warmup) {
                emit("mapped_stream_sync",
                     payload_bytes,
                     i - warmup,
                     us_since(t1, t0),
                     us_since(t2, t1),
                     0.0,
                     us_since(t2, t0),
                     0,
                     observed);
            }
        }

        for (int i = 0; i < total_iters; ++i) {
            clear_host();
            CUDA_CHECK(cudaStreamSynchronize(stream));
            const unsigned int value = 0x20000000u + unsigned(i);
            auto t0 = std::chrono::steady_clock::now();
            if (launch_mailbox(mailbox_fn, stream, host_dev, payload_words, cycles, value) != 0)
                return 1;
            auto t1 = std::chrono::steady_clock::now();
            unsigned long long polls = 0;
            volatile unsigned int *flag = reinterpret_cast<volatile unsigned int *>(host);
            while (*flag != value) {
                ++polls;
                if ((polls & 0x3fffull) == 0)
                    std::this_thread::yield();
            }
            auto t2 = std::chrono::steady_clock::now();
            CUDA_CHECK(cudaStreamSynchronize(stream));
            auto t3 = std::chrono::steady_clock::now();
            if (i >= warmup) {
                emit("mapped_cpu_poll",
                     payload_bytes,
                     i - warmup,
                     us_since(t1, t0),
                     us_since(t2, t1),
                     us_since(t3, t2),
                     us_since(t3, t0),
                     polls,
                     *flag);
            }
        }

        for (int i = 0; i < total_iters; ++i) {
            clear_host();
            CUDA_CHECK(cudaStreamSynchronize(stream));
            const unsigned int value = 0x30000000u + unsigned(i);
            auto t0 = std::chrono::steady_clock::now();
            if (launch_mailbox(mailbox_fn, stream, host_dev, payload_words, cycles, value) != 0)
                return 1;
            CUDA_CHECK(cudaEventRecord(event, stream));
            auto t1 = std::chrono::steady_clock::now();
            unsigned long long polls = 0;
            cudaError_t q = cudaErrorNotReady;
            do {
                q = cudaEventQuery(event);
                ++polls;
            } while (q == cudaErrorNotReady);
            if (q != cudaSuccess) {
                fprintf(stderr, "cudaEventQuery returned %s (%d)\n", cudaGetErrorString(q), int(q));
                return 1;
            }
            auto t2 = std::chrono::steady_clock::now();
            CUDA_CHECK(cudaStreamSynchronize(stream));
            auto t3 = std::chrono::steady_clock::now();
            unsigned int observed = *reinterpret_cast<volatile unsigned int *>(host);
            if (i >= warmup) {
                emit("mapped_event_query",
                     payload_bytes,
                     i - warmup,
                     us_since(t1, t0),
                     us_since(t2, t1),
                     us_since(t3, t2),
                     us_since(t3, t0),
                     polls,
                     observed);
            }
        }

        for (int i = 0; i < total_iters; ++i) {
            CUDA_CHECK(cudaStreamSynchronize(stream));
            auto t0 = std::chrono::steady_clock::now();
            if (launch_spin(spin_fn, stream, cycles) != 0)
                return 1;
            auto t1 = std::chrono::steady_clock::now();
            CUDA_CHECK(cudaMemcpyAsync(normal_host, dev_token, 4, cudaMemcpyDeviceToHost, stream));
            auto t2 = std::chrono::steady_clock::now();
            CUDA_CHECK(cudaStreamSynchronize(stream));
            auto t3 = std::chrono::steady_clock::now();
            unsigned int observed = *reinterpret_cast<volatile unsigned int *>(normal_host);
            if (i >= warmup) {
                emit("normal_d2h_after_spin",
                     payload_bytes,
                     i - warmup,
                     us_since(t1, t0),
                     us_since(t2, t1),
                     us_since(t3, t2),
                     us_since(t3, t0),
                     0,
                     observed);
            }
        }
    }

    fflush(out);
    fclose(out);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventDestroy(event));
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFreeHost(normal_host));
    CUDA_CHECK(cudaFree(dev_token));
    CUDA_CHECK(cudaFreeHost(host));
    CU_CHECK(cuModuleUnload(module));
    return 0;
}
