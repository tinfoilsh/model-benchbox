// Probe whether CC host-boundary waits depend on the kind of producer work.
//
// Build:
//   g++ -O2 -g cuda_producer_kind_boundary_probe.cpp \
//       -o cuda_producer_kind_boundary_probe \
//       -I/usr/local/cuda/include -L/usr/local/cuda/lib64 \
//       -lcudart -lcuda -Wl,-rpath,/usr/local/cuda/lib64
//
// Runtime env:
//   CPK_WORK_BYTES=4294967296
//   CPK_COPY_BYTES=4
//   CPK_KERNEL_CYCLES=2000000
//   CPK_REPEATS=5
//   CPK_WARMUP=1
//   CPK_OUT=/out/cuda_producer_kind_boundary.csv

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <string>
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

static const char *kSpinPtx = R"PTX(
.version 8.0
.target sm_90
.address_size 64

.visible .entry spin_kernel(
    .param .u64 cycles
)
{
    .reg .pred %p;
    .reg .u64 %cycles, %start, %now, %delta;
    ld.param.u64 %cycles, [cycles];
    mov.u64 %start, %clock64;
loop:
    mov.u64 %now, %clock64;
    sub.u64 %delta, %now, %start;
    setp.lt.u64 %p, %delta, %cycles;
    @%p bra loop;
    ret;
}
)PTX";

struct Timings {
    double work_api_us = 0.0;
    double boundary_api_us = 0.0;
    double consumer_sync_us = 0.0;
    double producer_sync_us = 0.0;
    double total_us = 0.0;
};

static unsigned long long env_ull(const char *name, unsigned long long fallback)
{
    const char *value = getenv(name);
    if (!value || !*value)
        return fallback;
    return strtoull(value, nullptr, 0);
}

static double us_since(std::chrono::steady_clock::time_point end,
                       std::chrono::steady_clock::time_point start)
{
    return std::chrono::duration<double, std::micro>(end - start).count();
}

static void write_header(FILE *out)
{
    fprintf(out,
            "scenario,iteration,work_api_us,boundary_api_us,"
            "consumer_sync_us,producer_sync_us,total_us\n");
}

static void write_row(FILE *out, const char *scenario, int iteration, const Timings &t)
{
    fprintf(out,
            "%s,%d,%.3f,%.3f,%.3f,%.3f,%.3f\n",
            scenario,
            iteration,
            t.work_api_us,
            t.boundary_api_us,
            t.consumer_sync_us,
            t.producer_sync_us,
            t.total_us);
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

int main()
{
    const size_t work_bytes = size_t(env_ull("CPK_WORK_BYTES", 4294967296ull));
    const size_t copy_bytes = size_t(env_ull("CPK_COPY_BYTES", 4ull));
    const unsigned long long kernel_cycles = env_ull("CPK_KERNEL_CYCLES", 2000000ull);
    const int repeats = int(env_ull("CPK_REPEATS", 5ull));
    const int warmup = int(env_ull("CPK_WARMUP", 1ull));
    const int total_iters = warmup + repeats;
    const char *out_path = getenv("CPK_OUT");
    if (!out_path || !*out_path)
        out_path = "/tmp/cuda_producer_kind_boundary.csv";

    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    CUDA_CHECK(cudaSetDevice(0));
    CU_CHECK(cuInit(0));

    CUmodule module = nullptr;
    CUfunction spin_fn = nullptr;
    CU_CHECK(cuModuleLoadData(&module, kSpinPtx));
    CU_CHECK(cuModuleGetFunction(&spin_fn, module, "spin_kernel"));

    void *dev_a = nullptr;
    void *dev_b = nullptr;
    void *dev_token = nullptr;
    void *host = nullptr;
    CUDA_CHECK(cudaMalloc(&dev_a, work_bytes + 4096));
    CUDA_CHECK(cudaMalloc(&dev_b, work_bytes + 4096));
    CUDA_CHECK(cudaMalloc(&dev_token, 4096));
    CUDA_CHECK(cudaHostAlloc(&host, 4096, 0));
    memset(host, 0, 4096);

    cudaStream_t producer = nullptr;
    cudaStream_t consumer = nullptr;
    cudaEvent_t event = nullptr;
    CUDA_CHECK(cudaStreamCreateWithFlags(&producer, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&consumer, cudaStreamNonBlocking));
    CUDA_CHECK(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
    CUDA_CHECK(cudaMemsetAsync(dev_a, 0xA5, work_bytes, producer));
    CUDA_CHECK(cudaMemsetAsync(dev_b, 0x5A, work_bytes, producer));
    CUDA_CHECK(cudaMemsetAsync(dev_token, 0x11, 4096, producer));
    CUDA_CHECK(cudaStreamSynchronize(producer));

    FILE *out = fopen(out_path, "w");
    if (!out) {
        perror("fopen(CPK_OUT)");
        return 1;
    }
    write_header(out);

    auto run_same_stream = [&](const char *scenario, const std::function<int()> &producer_work) -> int {
        for (int i = 0; i < total_iters; ++i) {
            CUDA_CHECK(cudaStreamSynchronize(producer));
            auto t0 = std::chrono::steady_clock::now();
            if (producer_work() != 0)
                return 1;
            auto t1 = std::chrono::steady_clock::now();
            CUDA_CHECK(cudaMemcpyAsync(host, dev_token, copy_bytes, cudaMemcpyDeviceToHost, producer));
            auto t2 = std::chrono::steady_clock::now();
            CUDA_CHECK(cudaStreamSynchronize(producer));
            auto t3 = std::chrono::steady_clock::now();
            if (i >= warmup) {
                Timings t;
                t.work_api_us = us_since(t1, t0);
                t.boundary_api_us = us_since(t2, t1);
                t.consumer_sync_us = us_since(t3, t2);
                t.total_us = us_since(t3, t0);
                write_row(out, scenario, i - warmup, t);
            }
        }
        return 0;
    };

    auto run_side_event_kernel = [&]() -> int {
        for (int i = 0; i < total_iters; ++i) {
            CUDA_CHECK(cudaStreamSynchronize(producer));
            CUDA_CHECK(cudaStreamSynchronize(consumer));
            auto t0 = std::chrono::steady_clock::now();
            if (launch_spin(spin_fn, producer, kernel_cycles) != 0)
                return 1;
            CUDA_CHECK(cudaEventRecord(event, producer));
            CUDA_CHECK(cudaStreamWaitEvent(consumer, event, 0));
            auto t1 = std::chrono::steady_clock::now();
            CUDA_CHECK(cudaMemcpyAsync(host, dev_token, copy_bytes, cudaMemcpyDeviceToHost, consumer));
            auto t2 = std::chrono::steady_clock::now();
            CUDA_CHECK(cudaStreamSynchronize(consumer));
            auto t3 = std::chrono::steady_clock::now();
            if (i >= warmup) {
                Timings t;
                t.work_api_us = us_since(t1, t0);
                t.boundary_api_us = us_since(t2, t1);
                t.consumer_sync_us = us_since(t3, t2);
                t.total_us = us_since(t3, t0);
                write_row(out, "side_event_kernel_d2h", i - warmup, t);
            }
        }
        return 0;
    };

    auto run_independent_kernel = [&]() -> int {
        for (int i = 0; i < total_iters; ++i) {
            CUDA_CHECK(cudaStreamSynchronize(producer));
            CUDA_CHECK(cudaStreamSynchronize(consumer));
            auto t0 = std::chrono::steady_clock::now();
            if (launch_spin(spin_fn, producer, kernel_cycles) != 0)
                return 1;
            auto t1 = std::chrono::steady_clock::now();
            CUDA_CHECK(cudaMemcpyAsync(host, dev_token, copy_bytes, cudaMemcpyDeviceToHost, consumer));
            auto t2 = std::chrono::steady_clock::now();
            CUDA_CHECK(cudaStreamSynchronize(consumer));
            auto t3 = std::chrono::steady_clock::now();
            CUDA_CHECK(cudaStreamSynchronize(producer));
            auto t4 = std::chrono::steady_clock::now();
            if (i >= warmup) {
                Timings t;
                t.work_api_us = us_since(t1, t0);
                t.boundary_api_us = us_since(t2, t1);
                t.consumer_sync_us = us_since(t3, t2);
                t.producer_sync_us = us_since(t4, t3);
                t.total_us = us_since(t4, t0);
                write_row(out, "independent_kernel_d2h", i - warmup, t);
            }
        }
        return 0;
    };

    if (run_same_stream("no_work_d2h", [&]() -> int { return 0; }) != 0)
        return 1;
    if (run_same_stream("memset_work_d2h", [&]() -> int {
            CUDA_CHECK(cudaMemsetAsync(dev_a, 0x3C, work_bytes, producer));
            return 0;
        }) != 0)
        return 1;
    if (run_same_stream("d2d_work_d2h", [&]() -> int {
            CUDA_CHECK(cudaMemcpyAsync(dev_b, dev_a, work_bytes, cudaMemcpyDeviceToDevice, producer));
            return 0;
        }) != 0)
        return 1;
    if (run_same_stream("kernel_work_d2h", [&]() -> int {
            return launch_spin(spin_fn, producer, kernel_cycles);
        }) != 0)
        return 1;
    if (run_side_event_kernel() != 0)
        return 1;
    if (run_independent_kernel() != 0)
        return 1;

    fflush(out);
    fclose(out);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventDestroy(event));
    CUDA_CHECK(cudaStreamDestroy(consumer));
    CUDA_CHECK(cudaStreamDestroy(producer));
    CUDA_CHECK(cudaFreeHost(host));
    CUDA_CHECK(cudaFree(dev_token));
    CUDA_CHECK(cudaFree(dev_b));
    CUDA_CHECK(cudaFree(dev_a));
    CU_CHECK(cuModuleUnload(module));
    return 0;
}
