// Probe whether pinned host-memory flavor changes CC ordered-boundary behavior.
//
// Build:
//   g++ -O2 -g cuda_host_memory_kind_boundary_probe.cpp \
//       -o cuda_host_memory_kind_boundary_probe \
//       -I/usr/local/cuda/include -L/usr/local/cuda/lib64 \
//       -lcudart -Wl,-rpath,/usr/local/cuda/lib64
//
// Runtime env:
//   CHK_WORK_BYTES=4294967296
//   CHK_COPY_BYTES=4,65536
//   CHK_REPEATS=5
//   CHK_WARMUP=1
//   CHK_OUT=/out/cuda_host_memory_kind_boundary.csv

#include <cuda_runtime_api.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <string>
#include <unistd.h>
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

struct HostMem {
    std::string name;
    void *ptr = nullptr;
    void *raw = nullptr;
    size_t bytes = 0;
    bool registered = false;
    bool free_host = false;
    bool free_runtime_host = false;
};

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

static size_t round_up(size_t value, size_t align)
{
    return ((value + align - 1) / align) * align;
}

static bool alloc_registered(HostMem &mem, size_t bytes, unsigned int flags)
{
    long page = sysconf(_SC_PAGESIZE);
    if (page <= 0)
        page = 4096;
    size_t rounded = round_up(bytes, size_t(page));
    void *raw = nullptr;
    if (posix_memalign(&raw, size_t(page), rounded) != 0)
        return false;
    memset(raw, 0x33, rounded);
    cudaError_t err = cudaHostRegister(raw, rounded, flags);
    if (err != cudaSuccess) {
        fprintf(stderr,
                "host kind %s failed cudaHostRegister: %s (%d)\n",
                mem.name.c_str(),
                cudaGetErrorString(err),
                int(err));
        free(raw);
        return false;
    }
    mem.ptr = raw;
    mem.raw = raw;
    mem.bytes = rounded;
    mem.registered = true;
    mem.free_host = true;
    return true;
}

static bool alloc_host_kind(HostMem &mem, size_t bytes)
{
    cudaError_t err = cudaSuccess;
    if (mem.name == "hostalloc_default") {
        err = cudaHostAlloc(&mem.ptr, bytes, cudaHostAllocDefault);
        mem.free_runtime_host = (err == cudaSuccess);
    } else if (mem.name == "hostalloc_portable") {
        err = cudaHostAlloc(&mem.ptr, bytes, cudaHostAllocPortable);
        mem.free_runtime_host = (err == cudaSuccess);
    } else if (mem.name == "hostalloc_mapped") {
        err = cudaHostAlloc(&mem.ptr, bytes, cudaHostAllocMapped);
        mem.free_runtime_host = (err == cudaSuccess);
    } else if (mem.name == "hostalloc_writecombined") {
        err = cudaHostAlloc(&mem.ptr, bytes, cudaHostAllocWriteCombined);
        mem.free_runtime_host = (err == cudaSuccess);
    } else if (mem.name == "hostalloc_mapped_writecombined") {
        err = cudaHostAlloc(&mem.ptr, bytes, cudaHostAllocMapped | cudaHostAllocWriteCombined);
        mem.free_runtime_host = (err == cudaSuccess);
    } else if (mem.name == "mallochost") {
        err = cudaMallocHost(&mem.ptr, bytes);
        mem.free_runtime_host = (err == cudaSuccess);
    } else if (mem.name == "register_default") {
        return alloc_registered(mem, bytes, cudaHostRegisterDefault);
    } else if (mem.name == "register_portable") {
        return alloc_registered(mem, bytes, cudaHostRegisterPortable);
    } else if (mem.name == "register_mapped") {
        return alloc_registered(mem, bytes, cudaHostRegisterMapped);
    } else {
        fprintf(stderr, "unknown host kind %s\n", mem.name.c_str());
        return false;
    }

    if (err != cudaSuccess) {
        fprintf(stderr,
                "host kind %s failed allocation: %s (%d)\n",
                mem.name.c_str(),
                cudaGetErrorString(err),
                int(err));
        return false;
    }
    mem.bytes = bytes;
    memset(mem.ptr, 0x22, bytes);
    return true;
}

static void free_host_kind(HostMem &mem)
{
    if (mem.registered && mem.ptr)
        cudaHostUnregister(mem.ptr);
    if (mem.free_runtime_host && mem.ptr)
        cudaFreeHost(mem.ptr);
    if (mem.free_host && mem.raw)
        free(mem.raw);
    mem.ptr = nullptr;
    mem.raw = nullptr;
}

static void write_header(FILE *out)
{
    fprintf(out,
            "host_kind,copy_bytes,op,with_work,iteration,work_api_us,"
            "boundary_api_us,sync_api_us,total_us\n");
}

int main()
{
    const size_t work_bytes = size_t(env_ull("CHK_WORK_BYTES", 4294967296ull));
    const int repeats = int(env_ull("CHK_REPEATS", 5ull));
    const int warmup = int(env_ull("CHK_WARMUP", 1ull));
    const int total_iters = warmup + repeats;
    const char *out_path = getenv("CHK_OUT");
    if (!out_path || !*out_path)
        out_path = "/tmp/cuda_host_memory_kind_boundary.csv";

    std::vector<size_t> copy_sizes = env_size_list("CHK_COPY_BYTES", "4,65536");
    if (copy_sizes.empty()) {
        fprintf(stderr, "no copy sizes\n");
        return 1;
    }
    const size_t max_copy = *std::max_element(copy_sizes.begin(), copy_sizes.end());
    const size_t host_bytes = std::max<size_t>(4096, max_copy);

    CUDA_CHECK(cudaSetDeviceFlags(cudaDeviceMapHost));
    CUDA_CHECK(cudaSetDevice(0));

    void *dev_work = nullptr;
    void *dev_copy = nullptr;
    CUDA_CHECK(cudaMalloc(&dev_work, std::max<size_t>(1, work_bytes)));
    CUDA_CHECK(cudaMalloc(&dev_copy, host_bytes));

    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaMemsetAsync(dev_work, 0xA5, work_bytes, stream));
    CUDA_CHECK(cudaMemsetAsync(dev_copy, 0x11, host_bytes, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    FILE *out = fopen(out_path, "w");
    if (!out) {
        perror("fopen(CHK_OUT)");
        return 1;
    }
    write_header(out);

    std::vector<std::string> host_kinds = {
        "hostalloc_default",
        "hostalloc_portable",
        "hostalloc_mapped",
        "hostalloc_writecombined",
        "hostalloc_mapped_writecombined",
        "mallochost",
        "register_default",
        "register_portable",
        "register_mapped",
    };

    for (const std::string &kind : host_kinds) {
        HostMem host;
        host.name = kind;
        if (!alloc_host_kind(host, host_bytes))
            continue;

        for (size_t copy_bytes : copy_sizes) {
            for (const char *op : {"d2h", "h2d"}) {
                for (int with_work : {0, 1}) {
                    for (int i = 0; i < total_iters; ++i) {
                        CUDA_CHECK(cudaStreamSynchronize(stream));
                        auto t0 = std::chrono::steady_clock::now();
                        if (with_work) {
                            CUDA_CHECK(cudaMemsetAsync(dev_work, 0x5A, work_bytes, stream));
                        }
                        auto t1 = std::chrono::steady_clock::now();
                        if (strcmp(op, "d2h") == 0) {
                            CUDA_CHECK(cudaMemcpyAsync(
                                host.ptr, dev_copy, copy_bytes, cudaMemcpyDeviceToHost, stream));
                        } else {
                            CUDA_CHECK(cudaMemcpyAsync(
                                dev_copy, host.ptr, copy_bytes, cudaMemcpyHostToDevice, stream));
                        }
                        auto t2 = std::chrono::steady_clock::now();
                        CUDA_CHECK(cudaStreamSynchronize(stream));
                        auto t3 = std::chrono::steady_clock::now();
                        if (i >= warmup) {
                            fprintf(out,
                                    "%s,%zu,%s,%d,%d,%.3f,%.3f,%.3f,%.3f\n",
                                    kind.c_str(),
                                    copy_bytes,
                                    op,
                                    with_work,
                                    i - warmup,
                                    us_since(t1, t0),
                                    us_since(t2, t1),
                                    us_since(t3, t2),
                                    us_since(t3, t0));
                        }
                    }
                }
            }
        }
        free_host_kind(host);
    }

    fflush(out);
    fclose(out);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(dev_copy));
    CUDA_CHECK(cudaFree(dev_work));
    return 0;
}
