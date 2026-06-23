// CUDA Driver API launch-tax probe that does not require the CUDA toolkit.
//
// The debug CVM images sometimes have libcuda and g++, but not nvcc or CUDA
// headers. This probe declares the small Driver API surface it needs, embeds a
// tiny PTX kernel, and loads libcuda dynamically.

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <utility>

using CUresult = int;
using CUdevice = int;
using CUdevice_attribute = int;
using CUjit_option = int;
using CUcontext = struct CUctx_st *;
using CUmodule = struct CUmod_st *;
using CUfunction = struct CUfunc_st *;
using CUstream = struct CUstream_st *;
using CUevent = struct CUevent_st *;
using CUgraph = struct CUgraph_st *;
using CUgraphNode = struct CUgraphNode_st *;
using CUgraphExec = struct CUgraphExec_st *;

constexpr CUresult CUDA_SUCCESS = 0;
constexpr unsigned int CU_EVENT_DISABLE_TIMING = 0x2;

struct CUDA_KERNEL_NODE_PARAMS {
    CUfunction func;
    unsigned int gridDimX;
    unsigned int gridDimY;
    unsigned int gridDimZ;
    unsigned int blockDimX;
    unsigned int blockDimY;
    unsigned int blockDimZ;
    unsigned int sharedMemBytes;
    void **kernelParams;
    void **extra;
};

static const char *kPtx = R"ptx(
.version 8.0
.target sm_70
.address_size 64

.visible .entry driver_empty()
{
    ret;
}
)ptx";

static double now_s()
{
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double>(clock::now().time_since_epoch()).count();
}

static int env_int(const char *name, int default_value)
{
    const char *value = std::getenv(name);
    if (!value || !*value)
        return default_value;
    char *end = nullptr;
    long parsed = std::strtol(value, &end, 0);
    return end && *end == '\0' ? static_cast<int>(parsed) : default_value;
}

static std::string env_str(const char *name, const char *default_value)
{
    const char *value = std::getenv(name);
    return value && *value ? std::string(value) : std::string(default_value);
}

class Driver {
  public:
    using cuInit_t = CUresult (*)(unsigned int);
    using cuDriverGetVersion_t = CUresult (*)(int *);
    using cuDeviceGet_t = CUresult (*)(CUdevice *, int);
    using cuDeviceGetName_t = CUresult (*)(char *, int, CUdevice);
    using cuDeviceGetAttribute_t = CUresult (*)(int *, CUdevice_attribute, CUdevice);
    using cuCtxCreate_t = CUresult (*)(CUcontext *, unsigned int, CUdevice);
    using cuCtxDestroy_t = CUresult (*)(CUcontext);
    using cuModuleLoadDataEx_t = CUresult (*)(CUmodule *, const void *, unsigned int,
                                              CUjit_option *, void **);
    using cuModuleGetFunction_t = CUresult (*)(CUfunction *, CUmodule, const char *);
    using cuModuleUnload_t = CUresult (*)(CUmodule);
    using cuStreamCreate_t = CUresult (*)(CUstream *, unsigned int);
    using cuStreamSynchronize_t = CUresult (*)(CUstream);
    using cuStreamDestroy_t = CUresult (*)(CUstream);
    using cuLaunchKernel_t = CUresult (*)(CUfunction, unsigned int, unsigned int,
                                          unsigned int, unsigned int, unsigned int,
                                          unsigned int, unsigned int, CUstream, void **,
                                          void **);
    using cuEventCreate_t = CUresult (*)(CUevent *, unsigned int);
    using cuEventRecord_t = CUresult (*)(CUevent, CUstream);
    using cuEventSynchronize_t = CUresult (*)(CUevent);
    using cuEventDestroy_t = CUresult (*)(CUevent);
    using cuGraphCreate_t = CUresult (*)(CUgraph *, unsigned int);
    using cuGraphAddKernelNode_t = CUresult (*)(CUgraphNode *, CUgraph,
                                                const CUgraphNode *, size_t,
                                                const CUDA_KERNEL_NODE_PARAMS *);
    using cuGraphInstantiateWithFlags_t = CUresult (*)(CUgraphExec *, CUgraph,
                                                       unsigned long long);
    using cuGraphLaunch_t = CUresult (*)(CUgraphExec, CUstream);
    using cuGraphExecDestroy_t = CUresult (*)(CUgraphExec);
    using cuGraphDestroy_t = CUresult (*)(CUgraph);
    using cuGetErrorName_t = CUresult (*)(CUresult, const char **);
    using cuGetErrorString_t = CUresult (*)(CUresult, const char **);

    cuInit_t cuInit = nullptr;
    cuDriverGetVersion_t cuDriverGetVersion = nullptr;
    cuDeviceGet_t cuDeviceGet = nullptr;
    cuDeviceGetName_t cuDeviceGetName = nullptr;
    cuDeviceGetAttribute_t cuDeviceGetAttribute = nullptr;
    cuCtxCreate_t cuCtxCreate = nullptr;
    cuCtxDestroy_t cuCtxDestroy = nullptr;
    cuModuleLoadDataEx_t cuModuleLoadDataEx = nullptr;
    cuModuleGetFunction_t cuModuleGetFunction = nullptr;
    cuModuleUnload_t cuModuleUnload = nullptr;
    cuStreamCreate_t cuStreamCreate = nullptr;
    cuStreamSynchronize_t cuStreamSynchronize = nullptr;
    cuStreamDestroy_t cuStreamDestroy = nullptr;
    cuLaunchKernel_t cuLaunchKernel = nullptr;
    cuEventCreate_t cuEventCreate = nullptr;
    cuEventRecord_t cuEventRecord = nullptr;
    cuEventSynchronize_t cuEventSynchronize = nullptr;
    cuEventDestroy_t cuEventDestroy = nullptr;
    cuGraphCreate_t cuGraphCreate = nullptr;
    cuGraphAddKernelNode_t cuGraphAddKernelNode = nullptr;
    cuGraphInstantiateWithFlags_t cuGraphInstantiateWithFlags = nullptr;
    cuGraphLaunch_t cuGraphLaunch = nullptr;
    cuGraphExecDestroy_t cuGraphExecDestroy = nullptr;
    cuGraphDestroy_t cuGraphDestroy = nullptr;
    cuGetErrorName_t cuGetErrorName = nullptr;
    cuGetErrorString_t cuGetErrorString = nullptr;

    void open()
    {
        lib_ = dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
        if (!lib_) {
            std::cerr << "dlopen(libcuda.so.1) failed: " << dlerror() << "\n";
            std::exit(1);
        }

        cuInit = sym<cuInit_t>("cuInit");
        cuDriverGetVersion = sym<cuDriverGetVersion_t>("cuDriverGetVersion");
        cuDeviceGet = sym<cuDeviceGet_t>("cuDeviceGet");
        cuDeviceGetName = sym<cuDeviceGetName_t>("cuDeviceGetName");
        cuDeviceGetAttribute = sym<cuDeviceGetAttribute_t>("cuDeviceGetAttribute");
        cuCtxCreate = sym<cuCtxCreate_t>("cuCtxCreate_v2", false);
        if (!cuCtxCreate)
            cuCtxCreate = sym<cuCtxCreate_t>("cuCtxCreate");
        cuCtxDestroy = sym<cuCtxDestroy_t>("cuCtxDestroy_v2", false);
        if (!cuCtxDestroy)
            cuCtxDestroy = sym<cuCtxDestroy_t>("cuCtxDestroy");
        cuModuleLoadDataEx = sym<cuModuleLoadDataEx_t>("cuModuleLoadDataEx");
        cuModuleGetFunction = sym<cuModuleGetFunction_t>("cuModuleGetFunction");
        cuModuleUnload = sym<cuModuleUnload_t>("cuModuleUnload");
        cuStreamCreate = sym<cuStreamCreate_t>("cuStreamCreate");
        cuStreamSynchronize = sym<cuStreamSynchronize_t>("cuStreamSynchronize");
        cuStreamDestroy = sym<cuStreamDestroy_t>("cuStreamDestroy_v2", false);
        if (!cuStreamDestroy)
            cuStreamDestroy = sym<cuStreamDestroy_t>("cuStreamDestroy");
        cuLaunchKernel = sym<cuLaunchKernel_t>("cuLaunchKernel");
        cuEventCreate = sym<cuEventCreate_t>("cuEventCreate");
        cuEventRecord = sym<cuEventRecord_t>("cuEventRecord");
        cuEventSynchronize = sym<cuEventSynchronize_t>("cuEventSynchronize");
        cuEventDestroy = sym<cuEventDestroy_t>("cuEventDestroy_v2", false);
        if (!cuEventDestroy)
            cuEventDestroy = sym<cuEventDestroy_t>("cuEventDestroy");
        cuGraphCreate = sym<cuGraphCreate_t>("cuGraphCreate");
        cuGraphAddKernelNode = sym<cuGraphAddKernelNode_t>("cuGraphAddKernelNode");
        cuGraphInstantiateWithFlags =
            sym<cuGraphInstantiateWithFlags_t>("cuGraphInstantiateWithFlags");
        cuGraphLaunch = sym<cuGraphLaunch_t>("cuGraphLaunch");
        cuGraphExecDestroy = sym<cuGraphExecDestroy_t>("cuGraphExecDestroy");
        cuGraphDestroy = sym<cuGraphDestroy_t>("cuGraphDestroy");
        cuGetErrorName = sym<cuGetErrorName_t>("cuGetErrorName");
        cuGetErrorString = sym<cuGetErrorString_t>("cuGetErrorString");
    }

    [[noreturn]] void die(CUresult result, const char *call, const char *file, int line)
    {
        const char *name = nullptr;
        const char *message = nullptr;
        if (cuGetErrorName)
            cuGetErrorName(result, &name);
        if (cuGetErrorString)
            cuGetErrorString(result, &message);
        std::cerr << "CUDA driver error from " << call << " at " << file << ":" << line
                  << ": " << (name ? name : "?") << " " << (message ? message : "")
                  << "\n";
        std::exit(1);
    }

  private:
    template <typename T>
    T sym(const char *name, bool required = true)
    {
        dlerror();
        void *handle = lib_;
        const char *use_preload = std::getenv("LTD_USE_PRELOAD_LAUNCH_WRAPPER");
        if (use_preload && std::strcmp(use_preload, "0") != 0 &&
            std::strcmp(name, "cuLaunchKernel") == 0)
            handle = RTLD_DEFAULT;
        void *ptr = dlsym(handle, name);
        const char *error = dlerror();
        if ((!ptr || error) && required) {
            std::cerr << "dlsym(" << name << ") failed";
            if (error)
                std::cerr << ": " << error;
            std::cerr << "\n";
            std::exit(1);
        }
        return reinterpret_cast<T>(ptr);
    }

    void *lib_ = nullptr;
};

static Driver g_cuda;

#define CHECK_CU(call)                                                         \
    do {                                                                       \
        CUresult _result = (call);                                             \
        if (_result != CUDA_SUCCESS)                                           \
            g_cuda.die(_result, #call, __FILE__, __LINE__);                    \
    } while (0)

struct TraceHooks {
    using hook_t = void (*)();
    hook_t reset = nullptr;
    hook_t dump = nullptr;

    TraceHooks()
    {
        reset = reinterpret_cast<hook_t>(dlsym(RTLD_DEFAULT, "evp_trace_reset"));
        dump = reinterpret_cast<hook_t>(dlsym(RTLD_DEFAULT, "evp_trace_dump"));
    }

    void begin(const char *label)
    {
        setenv("EVP_TRACE_LABEL", label, 1);
        if (reset)
            reset();
    }

    void end()
    {
        if (dump)
            dump();
    }
};

static bool want_mode(const std::string &mode, const char *name)
{
    return mode == "all" || mode == name;
}

static double measure_stream_sync_idle(int n, CUstream stream, TraceHooks &trace)
{
    CHECK_CU(g_cuda.cuStreamSynchronize(stream));
    trace.begin("stream_sync_idle");
    double t0 = now_s();
    for (int i = 0; i < n; ++i)
        CHECK_CU(g_cuda.cuStreamSynchronize(stream));
    double us = (now_s() - t0) / n * 1e6;
    trace.end();
    return us;
}

static double measure_event_record_pipelined(int n, CUstream stream, TraceHooks &trace)
{
    CUevent event = nullptr;
    CHECK_CU(g_cuda.cuEventCreate(&event, CU_EVENT_DISABLE_TIMING));
    CHECK_CU(g_cuda.cuStreamSynchronize(stream));

    trace.begin("event_record_pipelined");
    double t0 = now_s();
    for (int i = 0; i < n; ++i)
        CHECK_CU(g_cuda.cuEventRecord(event, stream));
    CHECK_CU(g_cuda.cuStreamSynchronize(stream));
    double us = (now_s() - t0) / n * 1e6;
    trace.end();

    CHECK_CU(g_cuda.cuEventDestroy(event));
    return us;
}

static double measure_event_record_roundtrip(int n, CUstream stream, TraceHooks &trace)
{
    CUevent event = nullptr;
    CHECK_CU(g_cuda.cuEventCreate(&event, CU_EVENT_DISABLE_TIMING));
    CHECK_CU(g_cuda.cuStreamSynchronize(stream));

    trace.begin("event_record_roundtrip");
    double t0 = now_s();
    for (int i = 0; i < n; ++i) {
        CHECK_CU(g_cuda.cuEventRecord(event, stream));
        CHECK_CU(g_cuda.cuEventSynchronize(event));
    }
    double us = (now_s() - t0) / n * 1e6;
    trace.end();

    CHECK_CU(g_cuda.cuEventDestroy(event));
    return us;
}

static double measure_driver_launch_pipelined(int n, CUfunction fn, CUstream stream,
                                              TraceHooks &trace)
{
    CHECK_CU(g_cuda.cuStreamSynchronize(stream));
    trace.begin("driver_launch_pipelined");
    double t0 = now_s();
    for (int i = 0; i < n; ++i)
        CHECK_CU(g_cuda.cuLaunchKernel(fn, 1, 1, 1, 1, 1, 1, 0, stream, nullptr, nullptr));
    CHECK_CU(g_cuda.cuStreamSynchronize(stream));
    double us = (now_s() - t0) / n * 1e6;
    trace.end();
    return us;
}

static std::pair<double, double> measure_driver_launch_issue_only(int n, CUfunction fn,
                                                                  CUstream stream,
                                                                  TraceHooks &trace)
{
    CHECK_CU(g_cuda.cuStreamSynchronize(stream));
    trace.begin("driver_launch_issue_only");
    double t0 = now_s();
    for (int i = 0; i < n; ++i)
        CHECK_CU(g_cuda.cuLaunchKernel(fn, 1, 1, 1, 1, 1, 1, 0, stream, nullptr, nullptr));
    double t1 = now_s();
    CHECK_CU(g_cuda.cuStreamSynchronize(stream));
    double t2 = now_s();
    trace.end();

    return {(t1 - t0) / n * 1e6, (t2 - t1) * 1e6};
}

static double measure_driver_launch_roundtrip(int n, CUfunction fn, CUstream stream,
                                              TraceHooks &trace)
{
    CHECK_CU(g_cuda.cuStreamSynchronize(stream));
    trace.begin("driver_launch_roundtrip");
    double t0 = now_s();
    for (int i = 0; i < n; ++i) {
        CHECK_CU(g_cuda.cuLaunchKernel(fn, 1, 1, 1, 1, 1, 1, 0, stream, nullptr, nullptr));
        CHECK_CU(g_cuda.cuStreamSynchronize(stream));
    }
    double us = (now_s() - t0) / n * 1e6;
    trace.end();
    return us;
}

static void build_graph(int k, CUfunction fn, CUgraph *graph, CUgraphExec *exec)
{
    CHECK_CU(g_cuda.cuGraphCreate(graph, 0));
    CUgraphNode previous = nullptr;

    for (int i = 0; i < k; ++i) {
        CUDA_KERNEL_NODE_PARAMS params{};
        params.func = fn;
        params.gridDimX = 1;
        params.gridDimY = 1;
        params.gridDimZ = 1;
        params.blockDimX = 1;
        params.blockDimY = 1;
        params.blockDimZ = 1;
        params.sharedMemBytes = 0;
        params.kernelParams = nullptr;
        params.extra = nullptr;

        CUgraphNode node = nullptr;
        if (previous)
            CHECK_CU(g_cuda.cuGraphAddKernelNode(&node, *graph, &previous, 1, &params));
        else
            CHECK_CU(g_cuda.cuGraphAddKernelNode(&node, *graph, nullptr, 0, &params));
        previous = node;
    }

    CHECK_CU(g_cuda.cuGraphInstantiateWithFlags(exec, *graph, 0));
}

static double measure_graph_build_us_per_kernel(int k, CUfunction fn, TraceHooks &trace)
{
    trace.begin("driver_graph_build");
    double t0 = now_s();
    CUgraph graph = nullptr;
    CUgraphExec exec = nullptr;
    build_graph(k, fn, &graph, &exec);
    double us = (now_s() - t0) / k * 1e6;
    trace.end();

    CHECK_CU(g_cuda.cuGraphExecDestroy(exec));
    CHECK_CU(g_cuda.cuGraphDestroy(graph));
    return us;
}

static std::pair<double, double> measure_graph_replay_us(int k, int reps, CUfunction fn,
                                                         CUstream stream, TraceHooks &trace)
{
    CUgraph graph = nullptr;
    CUgraphExec exec = nullptr;
    build_graph(k, fn, &graph, &exec);

    CHECK_CU(g_cuda.cuGraphLaunch(exec, stream));
    CHECK_CU(g_cuda.cuStreamSynchronize(stream));

    trace.begin("driver_graph_replay");
    double t0 = now_s();
    for (int i = 0; i < reps; ++i)
        CHECK_CU(g_cuda.cuGraphLaunch(exec, stream));
    CHECK_CU(g_cuda.cuStreamSynchronize(stream));
    double replay_us = (now_s() - t0) / reps * 1e6;
    trace.end();

    CHECK_CU(g_cuda.cuGraphExecDestroy(exec));
    CHECK_CU(g_cuda.cuGraphDestroy(graph));
    return {replay_us, replay_us / k};
}

class JsonMetrics {
  public:
    explicit JsonMetrics(std::ostream &out) : out_(out) {}

    void metric(const char *name, double value)
    {
        if (!first_)
            out_ << ",\n";
        out_ << "    \"" << name << "\": " << std::fixed << std::setprecision(3)
             << value;
        first_ = false;
    }

  private:
    std::ostream &out_;
    bool first_ = true;
};

int main()
{
    const int n = env_int("LTD_N", 5000);
    const int roundtrip_n = env_int("LTD_ROUNDTRIP_N", 500);
    const int event_n = env_int("LTD_EVENT_N", n);
    const int graph_k = env_int("LTD_GRAPH_K", 300);
    const int graph_reps = env_int("LTD_GRAPH_REPS", 200);
    const std::string mode = env_str("LTD_MODE", "all");

    g_cuda.open();
    CHECK_CU(g_cuda.cuInit(0));

    int driver_version = 0;
    CHECK_CU(g_cuda.cuDriverGetVersion(&driver_version));

    CUdevice device = 0;
    CHECK_CU(g_cuda.cuDeviceGet(&device, 0));
    char device_name[256]{};
    CHECK_CU(g_cuda.cuDeviceGetName(device_name, sizeof(device_name), device));

    CUcontext context = nullptr;
    CHECK_CU(g_cuda.cuCtxCreate(&context, 0, device));

    CUmodule module = nullptr;
    CHECK_CU(g_cuda.cuModuleLoadDataEx(&module, kPtx, 0, nullptr, nullptr));
    CUfunction fn = nullptr;
    CHECK_CU(g_cuda.cuModuleGetFunction(&fn, module, "driver_empty"));

    CUstream stream = nullptr;
    CHECK_CU(g_cuda.cuStreamCreate(&stream, 0));

    for (int i = 0; i < 100; ++i)
        CHECK_CU(g_cuda.cuLaunchKernel(fn, 1, 1, 1, 1, 1, 1, 0, stream, nullptr, nullptr));
    CHECK_CU(g_cuda.cuStreamSynchronize(stream));

    TraceHooks trace;

    std::cout << "{\n";
    std::cout << "  \"probe\": \"launch_tax_driver_ptx\",\n";
    std::cout << "  \"driver_version\": " << driver_version << ",\n";
    std::cout << "  \"device_name\": \"" << device_name << "\",\n";
    std::cout << "  \"mode\": \"" << mode << "\",\n";
    std::cout << "  \"n\": " << n << ",\n";
    std::cout << "  \"roundtrip_n\": " << roundtrip_n << ",\n";
    std::cout << "  \"event_n\": " << event_n << ",\n";
    std::cout << "  \"graph_k\": " << graph_k << ",\n";
    std::cout << "  \"graph_reps\": " << graph_reps << ",\n";
    std::cout << "  \"measurements_us\": {\n";

    JsonMetrics metrics(std::cout);
    if (want_mode(mode, "stream_sync_idle"))
        metrics.metric("stream_sync_idle", measure_stream_sync_idle(event_n, stream, trace));
    if (want_mode(mode, "event_record_pipelined"))
        metrics.metric("event_record_pipelined",
                       measure_event_record_pipelined(event_n, stream, trace));
    if (want_mode(mode, "event_record_roundtrip"))
        metrics.metric("event_record_roundtrip",
                       measure_event_record_roundtrip(roundtrip_n, stream, trace));
    if (want_mode(mode, "driver_launch_pipelined"))
        metrics.metric("driver_launch_pipelined",
                       measure_driver_launch_pipelined(n, fn, stream, trace));
    if (want_mode(mode, "driver_launch_issue_only")) {
        auto issue = measure_driver_launch_issue_only(n, fn, stream, trace);
        metrics.metric("driver_launch_issue_api", issue.first);
        metrics.metric("driver_launch_issue_final_sync", issue.second);
    }
    if (want_mode(mode, "driver_launch_roundtrip"))
        metrics.metric("driver_launch_roundtrip",
                       measure_driver_launch_roundtrip(roundtrip_n, fn, stream, trace));
    if (want_mode(mode, "driver_graph_build"))
        metrics.metric("driver_graph_build_per_kernel",
                       measure_graph_build_us_per_kernel(graph_k, fn, trace));
    if (want_mode(mode, "driver_graph_replay")) {
        auto replay = measure_graph_replay_us(graph_k, graph_reps, fn, stream, trace);
        metrics.metric("driver_graph_replay_per_replay", replay.first);
        metrics.metric("driver_graph_replay_per_kernel", replay.second);
    }

    std::cout << "\n  }\n";
    std::cout << "}\n";

    CHECK_CU(g_cuda.cuStreamDestroy(stream));
    CHECK_CU(g_cuda.cuModuleUnload(module));
    CHECK_CU(g_cuda.cuCtxDestroy(context));
    return 0;
}
