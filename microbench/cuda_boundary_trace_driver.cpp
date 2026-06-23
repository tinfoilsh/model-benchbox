// Small CUDA runtime driver for cuda_boundary_trace_preload.cpp.
//
// Build:
//   g++ -O2 -g cuda_boundary_trace_driver.cpp -o cuda_boundary_trace_driver \
//       -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -lcudart

#include <cuda_runtime_api.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>

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

static unsigned long long env_ull(const char *name, unsigned long long fallback)
{
    const char *value = getenv(name);
    if (!value || !*value)
        return fallback;
    return strtoull(value, nullptr, 0);
}

static void marker(const char *name, int iter)
{
    fprintf(stderr, "CBT_MARKER scenario=%s iter=%d\n", name, iter);
    fflush(stderr);
}

int main()
{
    const size_t work_bytes = size_t(env_ull("CBT_WORK_BYTES", 4294967296ull));
    const size_t copy_bytes = size_t(env_ull("CBT_COPY_BYTES", 4ull));
    const int repeats = int(env_ull("CBT_REPEATS", 3ull));
    const int warmup = int(env_ull("CBT_WARMUP", 1ull));
    const int total_iters = warmup + repeats;

    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    CUDA_CHECK(cudaSetDevice(0));

    void *dev_a = nullptr;
    void *dev_b = nullptr;
    void *host = nullptr;
    CUDA_CHECK(cudaMalloc(&dev_a, work_bytes + 4096));
    CUDA_CHECK(cudaMalloc(&dev_b, work_bytes + 4096));
    CUDA_CHECK(cudaHostAlloc(&host, 4096, 0));
    memset(host, 0, 4096);

    cudaStream_t producer = nullptr;
    cudaStream_t consumer = nullptr;
    cudaEvent_t event = nullptr;
    CUDA_CHECK(cudaStreamCreateWithFlags(&producer, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&consumer, cudaStreamNonBlocking));
    CUDA_CHECK(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
    CUDA_CHECK(cudaDeviceSynchronize());

    fprintf(stderr,
            "CBT_CONFIG device_count=%d work_bytes=%zu copy_bytes=%zu warmup=%d repeats=%d\n",
            device_count,
            work_bytes,
            copy_bytes,
            warmup,
            repeats);
    fflush(stderr);

    for (int i = 0; i < total_iters; ++i) {
        marker("same_stream_d2h", i);
        CUDA_CHECK(cudaStreamSynchronize(producer));
        CUDA_CHECK(cudaMemsetAsync(dev_a, 0x5A, work_bytes, producer));
        CUDA_CHECK(cudaMemcpyAsync(host, dev_a, copy_bytes, cudaMemcpyDeviceToHost, producer));
        CUDA_CHECK(cudaStreamSynchronize(producer));
    }

    for (int i = 0; i < total_iters; ++i) {
        marker("same_stream_h2d", i);
        CUDA_CHECK(cudaStreamSynchronize(producer));
        CUDA_CHECK(cudaMemsetAsync(dev_a, 0x5B, work_bytes, producer));
        CUDA_CHECK(cudaMemcpyAsync(dev_b, host, copy_bytes, cudaMemcpyHostToDevice, producer));
        CUDA_CHECK(cudaStreamSynchronize(producer));
    }

    for (int i = 0; i < total_iters; ++i) {
        marker("same_stream_d2d", i);
        CUDA_CHECK(cudaStreamSynchronize(producer));
        CUDA_CHECK(cudaMemsetAsync(dev_a, 0x5C, work_bytes, producer));
        CUDA_CHECK(cudaMemcpyAsync(dev_b, dev_a, copy_bytes, cudaMemcpyDeviceToDevice, producer));
        CUDA_CHECK(cudaStreamSynchronize(producer));
    }

    for (int i = 0; i < total_iters; ++i) {
        marker("side_stream_event_d2h", i);
        CUDA_CHECK(cudaStreamSynchronize(producer));
        CUDA_CHECK(cudaStreamSynchronize(consumer));
        CUDA_CHECK(cudaMemsetAsync(dev_a, 0x5D, work_bytes, producer));
        CUDA_CHECK(cudaEventRecord(event, producer));
        CUDA_CHECK(cudaStreamWaitEvent(consumer, event, 0));
        CUDA_CHECK(cudaMemcpyAsync(host, dev_a, copy_bytes, cudaMemcpyDeviceToHost, consumer));
        CUDA_CHECK(cudaStreamSynchronize(consumer));
    }

    for (int i = 0; i < total_iters; ++i) {
        marker("independent_stream_d2h", i);
        CUDA_CHECK(cudaStreamSynchronize(producer));
        CUDA_CHECK(cudaStreamSynchronize(consumer));
        CUDA_CHECK(cudaMemsetAsync(dev_a, 0x5E, work_bytes, producer));
        CUDA_CHECK(cudaMemcpyAsync(host, dev_b, copy_bytes, cudaMemcpyDeviceToHost, consumer));
        CUDA_CHECK(cudaStreamSynchronize(consumer));
        CUDA_CHECK(cudaStreamSynchronize(producer));
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventDestroy(event));
    CUDA_CHECK(cudaStreamDestroy(consumer));
    CUDA_CHECK(cudaStreamDestroy(producer));
    CUDA_CHECK(cudaFreeHost(host));
    CUDA_CHECK(cudaFree(dev_b));
    CUDA_CHECK(cudaFree(dev_a));
    return 0;
}
