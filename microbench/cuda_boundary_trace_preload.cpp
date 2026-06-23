// LD_PRELOAD tracer for CUDA boundary and ioctl timing.
//
// Build:
//   g++ -O2 -g -fPIC -shared cuda_boundary_trace_preload.cpp \
//       -o cuda_boundary_trace_preload.so \
//       -I/usr/local/cuda/include -ldl -pthread
//
// Runtime:
//   CBT_TRACE_FILE=/tmp/cuda_boundary_trace.csv

#include <cuda_runtime_api.h>

#include <dlfcn.h>
#include <pthread.h>
#include <stdarg.h>
#include <sys/ioctl.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <mutex>

static std::mutex g_log_mu;
static FILE *g_log = nullptr;
static std::atomic<unsigned long long> g_seq{0};
static thread_local bool g_in_trace = false;

static uint64_t now_ns()
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return uint64_t(ts.tv_sec) * 1000000000ull + uint64_t(ts.tv_nsec);
}

static long tid()
{
    return (long)(uintptr_t)pthread_self();
}

static FILE *log_file()
{
    std::lock_guard<std::mutex> lock(g_log_mu);
    if (g_log)
        return g_log;

    const char *path = getenv("CBT_TRACE_FILE");
    if (!path || !*path)
        path = "/tmp/cuda_boundary_trace.csv";

    g_log = fopen(path, "w");
    if (!g_log)
        return nullptr;

    fprintf(g_log, "seq,start_ns,tid,api,duration_ns,ret,detail\n");
    fflush(g_log);
    return g_log;
}

static void log_event(const char *api,
                      uint64_t start_ns,
                      uint64_t end_ns,
                      long long ret,
                      const char *detail)
{
    if (g_in_trace)
        return;
    g_in_trace = true;
    FILE *f = log_file();
    if (!f) {
        g_in_trace = false;
        return;
    }

    unsigned long long seq = ++g_seq;
    std::lock_guard<std::mutex> lock(g_log_mu);
    fprintf(f,
            "%llu,%llu,%ld,%s,%llu,%lld,\"%s\"\n",
            seq,
            (unsigned long long)start_ns,
            tid(),
            api,
            (unsigned long long)(end_ns - start_ns),
            ret,
            detail ? detail : "");
    fflush(f);
    g_in_trace = false;
}

template <typename Fn>
static Fn next_symbol(const char *name)
{
    void *sym = dlsym(RTLD_NEXT, name);
    if (!sym) {
        fprintf(stderr, "cuda_boundary_trace_preload: missing symbol %s: %s\n", name, dlerror());
        _exit(127);
    }
    return reinterpret_cast<Fn>(sym);
}

extern "C" int ioctl(int fd, unsigned long request, ...)
{
    using Fn = int (*)(int, unsigned long, void *);
    static Fn real_fn = next_symbol<Fn>("ioctl");

    va_list args;
    va_start(args, request);
    void *argp = va_arg(args, void *);
    va_end(args);

    char detail[128];
    snprintf(detail, sizeof(detail), "fd=%d request=0x%lx arg=%p", fd, request, argp);

    uint64_t start = now_ns();
    int ret = real_fn(fd, request, argp);
    uint64_t end = now_ns();
    log_event("ioctl", start, end, ret, detail);
    return ret;
}

static const char *syscall_name(long number)
{
    switch (number) {
#ifdef SYS_ioctl
        case SYS_ioctl:
            return "syscall.ioctl";
#endif
#ifdef SYS_futex
        case SYS_futex:
            return "syscall.futex";
#endif
#ifdef SYS_futex_waitv
        case SYS_futex_waitv:
            return "syscall.futex_waitv";
#endif
#ifdef SYS_nanosleep
        case SYS_nanosleep:
            return "syscall.nanosleep";
#endif
#ifdef SYS_clock_nanosleep
        case SYS_clock_nanosleep:
            return "syscall.clock_nanosleep";
#endif
#ifdef SYS_sched_yield
        case SYS_sched_yield:
            return "syscall.sched_yield";
#endif
#ifdef SYS_poll
        case SYS_poll:
            return "syscall.poll";
#endif
#ifdef SYS_ppoll
        case SYS_ppoll:
            return "syscall.ppoll";
#endif
#ifdef SYS_select
        case SYS_select:
            return "syscall.select";
#endif
#ifdef SYS_pselect6
        case SYS_pselect6:
            return "syscall.pselect6";
#endif
        default:
            return nullptr;
    }
}

extern "C" long syscall(long number, ...)
{
    using Fn = long (*)(long, long, long, long, long, long, long);
    static Fn real_fn = next_symbol<Fn>("syscall");

    va_list args;
    va_start(args, number);
    long a1 = va_arg(args, long);
    long a2 = va_arg(args, long);
    long a3 = va_arg(args, long);
    long a4 = va_arg(args, long);
    long a5 = va_arg(args, long);
    long a6 = va_arg(args, long);
    va_end(args);

    const char *name = g_in_trace ? nullptr : syscall_name(number);
    if (!name)
        return real_fn(number, a1, a2, a3, a4, a5, a6);

    char detail[192];
    snprintf(detail,
             sizeof(detail),
             "nr=%ld a1=0x%lx a2=0x%lx a3=0x%lx a4=0x%lx a5=0x%lx a6=0x%lx",
             number,
             a1,
             a2,
             a3,
             a4,
             a5,
             a6);

    uint64_t start = now_ns();
    long ret = real_fn(number, a1, a2, a3, a4, a5, a6);
    uint64_t end = now_ns();
    log_event(name, start, end, ret, detail);
    return ret;
}

extern "C" cudaError_t cudaMemcpyAsync(void *dst,
                                        const void *src,
                                        size_t count,
                                        enum cudaMemcpyKind kind,
                                        cudaStream_t stream)
{
    using Fn = cudaError_t (*)(void *, const void *, size_t, enum cudaMemcpyKind, cudaStream_t);
    static Fn real_fn = next_symbol<Fn>("cudaMemcpyAsync");

    char detail[160];
    snprintf(detail, sizeof(detail), "count=%zu kind=%d dst=%p src=%p stream=%p",
             count, int(kind), dst, src, stream);

    uint64_t start = now_ns();
    cudaError_t ret = real_fn(dst, src, count, kind, stream);
    uint64_t end = now_ns();
    log_event("cudaMemcpyAsync", start, end, (long long)ret, detail);
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

    char detail[160];
    snprintf(detail, sizeof(detail), "count=%zu kind=%d dst=%p src=%p stream=%p",
             count, int(kind), dst, src, stream);

    uint64_t start = now_ns();
    cudaError_t ret = real_fn(dst, src, count, kind, stream);
    uint64_t end = now_ns();
    log_event("cudaMemcpyAsync_ptsz", start, end, (long long)ret, detail);
    return ret;
}

extern "C" cudaError_t cudaMemsetAsync(void *devPtr,
                                        int value,
                                        size_t count,
                                        cudaStream_t stream)
{
    using Fn = cudaError_t (*)(void *, int, size_t, cudaStream_t);
    static Fn real_fn = next_symbol<Fn>("cudaMemsetAsync");

    char detail[128];
    snprintf(detail, sizeof(detail), "count=%zu value=%d ptr=%p stream=%p",
             count, value, devPtr, stream);

    uint64_t start = now_ns();
    cudaError_t ret = real_fn(devPtr, value, count, stream);
    uint64_t end = now_ns();
    log_event("cudaMemsetAsync", start, end, (long long)ret, detail);
    return ret;
}

extern "C" cudaError_t cudaMemsetAsync_ptsz(void *devPtr,
                                             int value,
                                             size_t count,
                                             cudaStream_t stream)
{
    using Fn = cudaError_t (*)(void *, int, size_t, cudaStream_t);
    static Fn real_fn = next_symbol<Fn>("cudaMemsetAsync_ptsz");

    char detail[128];
    snprintf(detail, sizeof(detail), "count=%zu value=%d ptr=%p stream=%p",
             count, value, devPtr, stream);

    uint64_t start = now_ns();
    cudaError_t ret = real_fn(devPtr, value, count, stream);
    uint64_t end = now_ns();
    log_event("cudaMemsetAsync_ptsz", start, end, (long long)ret, detail);
    return ret;
}

extern "C" cudaError_t cudaStreamSynchronize(cudaStream_t stream)
{
    using Fn = cudaError_t (*)(cudaStream_t);
    static Fn real_fn = next_symbol<Fn>("cudaStreamSynchronize");

    char detail[80];
    snprintf(detail, sizeof(detail), "stream=%p", stream);

    uint64_t start = now_ns();
    cudaError_t ret = real_fn(stream);
    uint64_t end = now_ns();
    log_event("cudaStreamSynchronize", start, end, (long long)ret, detail);
    return ret;
}

extern "C" cudaError_t cudaDeviceSynchronize()
{
    using Fn = cudaError_t (*)();
    static Fn real_fn = next_symbol<Fn>("cudaDeviceSynchronize");

    uint64_t start = now_ns();
    cudaError_t ret = real_fn();
    uint64_t end = now_ns();
    log_event("cudaDeviceSynchronize", start, end, (long long)ret, "");
    return ret;
}

extern "C" cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream)
{
    using Fn = cudaError_t (*)(cudaEvent_t, cudaStream_t);
    static Fn real_fn = next_symbol<Fn>("cudaEventRecord");

    char detail[96];
    snprintf(detail, sizeof(detail), "event=%p stream=%p", event, stream);

    uint64_t start = now_ns();
    cudaError_t ret = real_fn(event, stream);
    uint64_t end = now_ns();
    log_event("cudaEventRecord", start, end, (long long)ret, detail);
    return ret;
}

extern "C" cudaError_t cudaStreamWaitEvent(cudaStream_t stream,
                                            cudaEvent_t event,
                                            unsigned int flags)
{
    using Fn = cudaError_t (*)(cudaStream_t, cudaEvent_t, unsigned int);
    static Fn real_fn = next_symbol<Fn>("cudaStreamWaitEvent");

    char detail[128];
    snprintf(detail, sizeof(detail), "stream=%p event=%p flags=%u", stream, event, flags);

    uint64_t start = now_ns();
    cudaError_t ret = real_fn(stream, event, flags);
    uint64_t end = now_ns();
    log_event("cudaStreamWaitEvent", start, end, (long long)ret, detail);
    return ret;
}
