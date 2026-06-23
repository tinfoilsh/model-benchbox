// LD_PRELOAD tracer for CUDA launch/sync APIs.
//
// Build:
//   g++ -O2 -g -fPIC -shared launch_trace_preload.cpp -o launch_trace_preload.so \
//       -I/usr/local/cuda/include -ldl -pthread
//
// Runtime knobs:
//   LT_TRACE_FILE=/tmp/launch_trace.csv
//   LT_TRACE_STACKS=1
//   LT_TRACE_FIRST_N_STACKS=32
//   LT_TRACE_MIN_STACK_US=0

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <dlfcn.h>
#include <execinfo.h>
#include <pthread.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>

static std::mutex g_log_mu;
static FILE *g_log = nullptr;
static std::atomic<unsigned long long> g_seq{0};
static int g_trace_stacks = -1;
static int g_first_n_stacks = -1;
static double g_min_stack_us = -1.0;

static uint64_t now_ns()
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return uint64_t(ts.tv_sec) * 1000000000ull + uint64_t(ts.tv_nsec);
}

static long tid()
{
    return syscall(SYS_gettid);
}

static int env_int(const char *name, int default_value)
{
    const char *v = getenv(name);
    if (!v || !*v)
        return default_value;
    return atoi(v);
}

static double env_double(const char *name, double default_value)
{
    const char *v = getenv(name);
    if (!v || !*v)
        return default_value;
    return atof(v);
}

static void init_config()
{
    if (g_trace_stacks >= 0)
        return;
    g_trace_stacks = env_int("LT_TRACE_STACKS", 0);
    g_first_n_stacks = env_int("LT_TRACE_FIRST_N_STACKS", 32);
    g_min_stack_us = env_double("LT_TRACE_MIN_STACK_US", 0.0);
}

static FILE *log_file()
{
    std::lock_guard<std::mutex> lock(g_log_mu);
    if (g_log)
        return g_log;
    const char *path = getenv("LT_TRACE_FILE");
    if (!path || !*path)
        path = "/tmp/launch_trace.csv";
    g_log = fopen(path, "w");
    if (!g_log)
        return nullptr;
    fprintf(g_log, "seq,tid,api,duration_ns,ret,stack\n");
    fflush(g_log);
    return g_log;
}

static std::string scrub(const char *s)
{
    std::string out = s ? s : "";
    for (char &c : out) {
        if (c == ',' || c == '\n' || c == '\r')
            c = ' ';
        else if (c == '"')
            c = '\'';
    }
    return out;
}

static std::string stack_string()
{
    void *frames[64];
    int n = backtrace(frames, 64);
    char **symbols = backtrace_symbols(frames, n);
    std::string out;
    if (symbols) {
        for (int i = 2; i < n; ++i) {
            if (i > 2)
                out += " | ";
            out += scrub(symbols[i]);
        }
        free(symbols);
    }
    return out;
}

static void log_event(const char *api, uint64_t start_ns, uint64_t end_ns, long long ret)
{
    init_config();
    unsigned long long seq = ++g_seq;
    double duration_us = double(end_ns - start_ns) / 1000.0;
    bool want_stack = g_trace_stacks &&
                      int(seq) <= g_first_n_stacks &&
                      duration_us >= g_min_stack_us;
    std::string stack = want_stack ? stack_string() : "";

    FILE *f = log_file();
    if (!f)
        return;

    std::lock_guard<std::mutex> lock(g_log_mu);
    fprintf(f, "%llu,%ld,%s,%llu,%lld,\"%s\"\n",
            seq,
            tid(),
            api,
            (unsigned long long)(end_ns - start_ns),
            ret,
            stack.c_str());
    fflush(f);
}

template <typename Fn>
static Fn next_symbol(const char *name)
{
    void *sym = dlsym(RTLD_NEXT, name);
    if (!sym) {
        fprintf(stderr, "launch_trace_preload: missing symbol %s: %s\n", name, dlerror());
        _exit(127);
    }
    return reinterpret_cast<Fn>(sym);
}

extern "C" cudaError_t cudaLaunchKernel(const void *func,
                                         dim3 gridDim,
                                         dim3 blockDim,
                                         void **args,
                                         size_t sharedMem,
                                         cudaStream_t stream)
{
    using Fn = cudaError_t (*)(const void *, dim3, dim3, void **, size_t, cudaStream_t);
    static Fn real_fn = next_symbol<Fn>("cudaLaunchKernel");
    uint64_t start = now_ns();
    cudaError_t ret = real_fn(func, gridDim, blockDim, args, sharedMem, stream);
    uint64_t end = now_ns();
    log_event("cudaLaunchKernel", start, end, (long long)ret);
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
    using Fn = CUresult (*)(CUfunction,
                            unsigned int,
                            unsigned int,
                            unsigned int,
                            unsigned int,
                            unsigned int,
                            unsigned int,
                            unsigned int,
                            CUstream,
                            void **,
                            void **);
    static Fn real_fn = next_symbol<Fn>("cuLaunchKernel");
    uint64_t start = now_ns();
    CUresult ret = real_fn(f,
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
    uint64_t end = now_ns();
    log_event("cuLaunchKernel", start, end, (long long)ret);
    return ret;
}

extern "C" CUresult cuLaunchKernel_ptsz(CUfunction f,
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
    using Fn = CUresult (*)(CUfunction,
                            unsigned int,
                            unsigned int,
                            unsigned int,
                            unsigned int,
                            unsigned int,
                            unsigned int,
                            unsigned int,
                            CUstream,
                            void **,
                            void **);
    static Fn real_fn = next_symbol<Fn>("cuLaunchKernel_ptsz");
    uint64_t start = now_ns();
    CUresult ret = real_fn(f,
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
    uint64_t end = now_ns();
    log_event("cuLaunchKernel_ptsz", start, end, (long long)ret);
    return ret;
}

extern "C" cudaError_t cudaDeviceSynchronize()
{
    using Fn = cudaError_t (*)();
    static Fn real_fn = next_symbol<Fn>("cudaDeviceSynchronize");
    uint64_t start = now_ns();
    cudaError_t ret = real_fn();
    uint64_t end = now_ns();
    log_event("cudaDeviceSynchronize", start, end, (long long)ret);
    return ret;
}

extern "C" cudaError_t cudaStreamSynchronize(cudaStream_t stream)
{
    using Fn = cudaError_t (*)(cudaStream_t);
    static Fn real_fn = next_symbol<Fn>("cudaStreamSynchronize");
    uint64_t start = now_ns();
    cudaError_t ret = real_fn(stream);
    uint64_t end = now_ns();
    log_event("cudaStreamSynchronize", start, end, (long long)ret);
    return ret;
}

extern "C" CUresult cuStreamSynchronize(CUstream stream)
{
    using Fn = CUresult (*)(CUstream);
    static Fn real_fn = next_symbol<Fn>("cuStreamSynchronize");
    uint64_t start = now_ns();
    CUresult ret = real_fn(stream);
    uint64_t end = now_ns();
    log_event("cuStreamSynchronize", start, end, (long long)ret);
    return ret;
}

extern "C" CUresult cuStreamSynchronize_ptsz(CUstream stream)
{
    using Fn = CUresult (*)(CUstream);
    static Fn real_fn = next_symbol<Fn>("cuStreamSynchronize_ptsz");
    uint64_t start = now_ns();
    CUresult ret = real_fn(stream);
    uint64_t end = now_ns();
    log_event("cuStreamSynchronize_ptsz", start, end, (long long)ret);
    return ret;
}

extern "C" CUresult cuCtxSynchronize()
{
    using Fn = CUresult (*)();
    static Fn real_fn = next_symbol<Fn>("cuCtxSynchronize");
    uint64_t start = now_ns();
    CUresult ret = real_fn();
    uint64_t end = now_ns();
    log_event("cuCtxSynchronize", start, end, (long long)ret);
    return ret;
}

extern "C" cudaError_t cudaGraphLaunch(cudaGraphExec_t graphExec, cudaStream_t stream)
{
    using Fn = cudaError_t (*)(cudaGraphExec_t, cudaStream_t);
    static Fn real_fn = next_symbol<Fn>("cudaGraphLaunch");
    uint64_t start = now_ns();
    cudaError_t ret = real_fn(graphExec, stream);
    uint64_t end = now_ns();
    log_event("cudaGraphLaunch", start, end, (long long)ret);
    return ret;
}

extern "C" CUresult cuGraphLaunch(CUgraphExec hGraphExec, CUstream hStream)
{
    using Fn = CUresult (*)(CUgraphExec, CUstream);
    static Fn real_fn = next_symbol<Fn>("cuGraphLaunch");
    uint64_t start = now_ns();
    CUresult ret = real_fn(hGraphExec, hStream);
    uint64_t end = now_ns();
    log_event("cuGraphLaunch", start, end, (long long)ret);
    return ret;
}

extern "C" CUresult cuGraphLaunch_ptsz(CUgraphExec hGraphExec, CUstream hStream)
{
    using Fn = CUresult (*)(CUgraphExec, CUstream);
    static Fn real_fn = next_symbol<Fn>("cuGraphLaunch_ptsz");
    uint64_t start = now_ns();
    CUresult ret = real_fn(hGraphExec, hStream);
    uint64_t end = now_ns();
    log_event("cuGraphLaunch_ptsz", start, end, (long long)ret);
    return ret;
}

extern "C" CUresult cuStreamBeginCapture(CUstream hStream, CUstreamCaptureMode mode)
{
    using Fn = CUresult (*)(CUstream, CUstreamCaptureMode);
    static Fn real_fn = next_symbol<Fn>("cuStreamBeginCapture");
    uint64_t start = now_ns();
    CUresult ret = real_fn(hStream, mode);
    uint64_t end = now_ns();
    log_event("cuStreamBeginCapture", start, end, (long long)ret);
    return ret;
}

extern "C" CUresult cuStreamBeginCapture_ptsz(CUstream hStream, CUstreamCaptureMode mode)
{
    using Fn = CUresult (*)(CUstream, CUstreamCaptureMode);
    static Fn real_fn = next_symbol<Fn>("cuStreamBeginCapture_ptsz");
    uint64_t start = now_ns();
    CUresult ret = real_fn(hStream, mode);
    uint64_t end = now_ns();
    log_event("cuStreamBeginCapture_ptsz", start, end, (long long)ret);
    return ret;
}

extern "C" CUresult cuStreamEndCapture(CUstream hStream, CUgraph *phGraph)
{
    using Fn = CUresult (*)(CUstream, CUgraph *);
    static Fn real_fn = next_symbol<Fn>("cuStreamEndCapture");
    uint64_t start = now_ns();
    CUresult ret = real_fn(hStream, phGraph);
    uint64_t end = now_ns();
    log_event("cuStreamEndCapture", start, end, (long long)ret);
    return ret;
}

extern "C" CUresult cuStreamEndCapture_ptsz(CUstream hStream, CUgraph *phGraph)
{
    using Fn = CUresult (*)(CUstream, CUgraph *);
    static Fn real_fn = next_symbol<Fn>("cuStreamEndCapture_ptsz");
    uint64_t start = now_ns();
    CUresult ret = real_fn(hStream, phGraph);
    uint64_t end = now_ns();
    log_event("cuStreamEndCapture_ptsz", start, end, (long long)ret);
    return ret;
}

extern "C" CUresult cuGraphInstantiateWithFlags(CUgraphExec *phGraphExec,
                                                 CUgraph hGraph,
                                                 unsigned long long flags)
{
    using Fn = CUresult (*)(CUgraphExec *, CUgraph, unsigned long long);
    static Fn real_fn = next_symbol<Fn>("cuGraphInstantiateWithFlags");
    uint64_t start = now_ns();
    CUresult ret = real_fn(phGraphExec, hGraph, flags);
    uint64_t end = now_ns();
    log_event("cuGraphInstantiateWithFlags", start, end, (long long)ret);
    return ret;
}

extern "C" CUresult cuGetProcAddress_v2(const char *symbol,
                                         void **pfn,
                                         int cudaVersion,
                                         cuuint64_t flags,
                                         CUdriverProcAddressQueryResult *symbolStatus)
{
    using Fn = CUresult (*)(const char *,
                            void **,
                            int,
                            cuuint64_t,
                            CUdriverProcAddressQueryResult *);
    static Fn real_fn = next_symbol<Fn>("cuGetProcAddress_v2");
    uint64_t start = now_ns();
    CUresult ret = real_fn(symbol, pfn, cudaVersion, flags, symbolStatus);
    uint64_t end = now_ns();

    std::string api = "cuGetProcAddress:";
    api += symbol ? symbol : "(null)";
    log_event(api.c_str(), start, end, (long long)ret);

    if (ret == CUDA_SUCCESS && pfn && *pfn && symbol) {
        if (strcmp(symbol, "cuLaunchKernel") == 0)
            *pfn = reinterpret_cast<void *>(&cuLaunchKernel);
        else if (strcmp(symbol, "cuLaunchKernel_ptsz") == 0)
            *pfn = reinterpret_cast<void *>(&cuLaunchKernel_ptsz);
        else if (strcmp(symbol, "cuStreamSynchronize") == 0)
            *pfn = reinterpret_cast<void *>(&cuStreamSynchronize);
        else if (strcmp(symbol, "cuStreamSynchronize_ptsz") == 0)
            *pfn = reinterpret_cast<void *>(&cuStreamSynchronize_ptsz);
        else if (strcmp(symbol, "cuGraphLaunch") == 0)
            *pfn = reinterpret_cast<void *>(&cuGraphLaunch);
        else if (strcmp(symbol, "cuGraphLaunch_ptsz") == 0)
            *pfn = reinterpret_cast<void *>(&cuGraphLaunch_ptsz);
        else if (strcmp(symbol, "cuStreamBeginCapture") == 0 || strcmp(symbol, "cuStreamBeginCapture_v2") == 0)
            *pfn = reinterpret_cast<void *>(&cuStreamBeginCapture_v2);
        else if (strcmp(symbol, "cuStreamBeginCapture_ptsz") == 0)
            *pfn = reinterpret_cast<void *>(&cuStreamBeginCapture_ptsz);
        else if (strcmp(symbol, "cuStreamEndCapture") == 0)
            *pfn = reinterpret_cast<void *>(&cuStreamEndCapture);
        else if (strcmp(symbol, "cuStreamEndCapture_ptsz") == 0)
            *pfn = reinterpret_cast<void *>(&cuStreamEndCapture_ptsz);
        else if (strcmp(symbol, "cuGraphInstantiateWithFlags") == 0)
            *pfn = reinterpret_cast<void *>(&cuGraphInstantiateWithFlags);
    }

    return ret;
}

__attribute__((destructor)) static void fini()
{
    std::lock_guard<std::mutex> lock(g_log_mu);
    if (g_log) {
        fflush(g_log);
        fclose(g_log);
        g_log = nullptr;
    }
}
