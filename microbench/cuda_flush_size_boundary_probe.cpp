// Probe D2H flush-size scaling for CC ordered host-visible boundaries.
//
// Build:
//   g++ -O2 -g cuda_flush_size_boundary_probe.cpp \
//       -o cuda_flush_size_boundary_probe \
//       -I/usr/local/cuda/include -L/usr/local/cuda/lib64 \
//       -lcudart -Wl,-rpath,/usr/local/cuda/lib64
//
// Runtime env:
//   CFS_WORK_BYTES=4294967296
//   CFS_COPY_BYTES=4,64,256,1024,4096,16384,65536,262144,1048576,4194304,16777216
//   CFS_REPEATS=5
//   CFS_WARMUP=1
//   CFS_OUT=/out/cuda_flush_size_boundary.csv

#include <cuda_runtime_api.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sstream>
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
        if (item.empty())
            continue;
        out.push_back(size_t(strtoull(item.c_str(), nullptr, 0)));
    }
    return out;
}

static double us_since(std::chrono::steady_clock::time_point end,
                       std::chrono::steady_clock::time_point start)
{
    return std::chrono::duration<double, std::micro>(end - start).count();
}

static void write_header(FILE *out)
{
    fprintf(out,
            "scenario,copy_bytes,iteration,work_api_us,boundary_api_us,"
            "sync_api_us,total_us\n");
}

int main()
{
    const size_t work_bytes = size_t(env_ull("CFS_WORK_BYTES", 4294967296ull));
    const int repeats = int(env_ull("CFS_REPEATS", 5ull));
    const int warmup = int(env_ull("CFS_WARMUP", 1ull));
    const int total_iters = warmup + repeats;
    const char *out_path = getenv("CFS_OUT");
    if (!out_path || !*out_path)
        out_path = "/tmp/cuda_flush_size_boundary.csv";

    std::vector<size_t> copy_sizes = env_size_list(
        "CFS_COPY_BYTES",
        "4,64,256,1024,4096,16384,65536,262144,1048576,4194304,16777216");
    if (copy_sizes.empty()) {
        fprintf(stderr, "no copy sizes\n");
        return 1;
    }
    const size_t max_copy = *std::max_element(copy_sizes.begin(), copy_sizes.end());

    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    CUDA_CHECK(cudaSetDevice(0));

    void *dev_work = nullptr;
    void *dev_copy = nullptr;
    void *host = nullptr;
    CUDA_CHECK(cudaMalloc(&dev_work, std::max<size_t>(1, work_bytes)));
    CUDA_CHECK(cudaMalloc(&dev_copy, std::max<size_t>(4096, max_copy)));
    CUDA_CHECK(cudaHostAlloc(&host, std::max<size_t>(4096, max_copy), 0));
    memset(host, 0, std::max<size_t>(4096, max_copy));

    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaMemsetAsync(dev_work, 0xA5, work_bytes, stream));
    CUDA_CHECK(cudaMemsetAsync(dev_copy, 0x11, std::max<size_t>(4096, max_copy), stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    FILE *out = fopen(out_path, "w");
    if (!out) {
        perror("fopen(CFS_OUT)");
        return 1;
    }
    write_header(out);

    auto run_case = [&](const char *scenario, size_t copy_bytes, bool with_work) -> int {
        for (int i = 0; i < total_iters; ++i) {
            CUDA_CHECK(cudaStreamSynchronize(stream));
            auto t0 = std::chrono::steady_clock::now();
            if (with_work) {
                CUDA_CHECK(cudaMemsetAsync(dev_work, 0x5A, work_bytes, stream));
            }
            auto t1 = std::chrono::steady_clock::now();
            CUDA_CHECK(cudaMemcpyAsync(host, dev_copy, copy_bytes, cudaMemcpyDeviceToHost, stream));
            auto t2 = std::chrono::steady_clock::now();
            CUDA_CHECK(cudaStreamSynchronize(stream));
            auto t3 = std::chrono::steady_clock::now();
            if (i >= warmup) {
                fprintf(out,
                        "%s,%zu,%d,%.3f,%.3f,%.3f,%.3f\n",
                        scenario,
                        copy_bytes,
                        i - warmup,
                        us_since(t1, t0),
                        us_since(t2, t1),
                        us_since(t3, t2),
                        us_since(t3, t0));
            }
        }
        return 0;
    };

    for (size_t copy_bytes : copy_sizes) {
        if (run_case("no_work_d2h", copy_bytes, false) != 0)
            return 1;
        if (run_case("after_work_d2h", copy_bytes, true) != 0)
            return 1;
    }

    fflush(out);
    fclose(out);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFreeHost(host));
    CUDA_CHECK(cudaFree(dev_copy));
    CUDA_CHECK(cudaFree(dev_work));
    return 0;
}
