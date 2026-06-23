// Gated CUDA launch loop for kernel ftrace experiments.
//
// The program initializes CUDA, warms up, writes LPG_READY_FILE, then waits for
// LPG_GO_FILE before running a tight runtime or Driver API launch loop. This
// lets an external script enable kernel tracing only for steady-state launches.

#include <cuda.h>
#include <cuda_runtime.h>

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <fstream>
#include <iostream>
#include <string>
#include <thread>

#define CHECK_CUDA(call)                                                       \
    do {                                                                       \
        cudaError_t _err = (call);                                             \
        if (_err != cudaSuccess) {                                             \
            std::cerr << "CUDA error " << cudaGetErrorString(_err)            \
                      << " at " << __FILE__ << ":" << __LINE__ << "\n";     \
            std::exit(1);                                                      \
        }                                                                      \
    } while (0)

#define CHECK_CU(call)                                                         \
    do {                                                                       \
        CUresult _err = (call);                                                \
        if (_err != CUDA_SUCCESS) {                                            \
            const char *name = nullptr;                                        \
            const char *msg = nullptr;                                         \
            cuGetErrorName(_err, &name);                                       \
            cuGetErrorString(_err, &msg);                                      \
            std::cerr << "CUDA driver error " << (name ? name : "?")         \
                      << " " << (msg ? msg : "")                             \
                      << " at " << __FILE__ << ":" << __LINE__ << "\n";     \
            std::exit(1);                                                      \
        }                                                                      \
    } while (0)

static int env_int(const char *name, int default_value)
{
    const char *value = std::getenv(name);
    if (!value || !*value)
        return default_value;
    return std::atoi(value);
}

static std::string env_string(const char *name, const char *default_value)
{
    const char *value = std::getenv(name);
    if (!value || !*value)
        return default_value;
    return value;
}

static bool file_exists(const std::string &path)
{
    std::ifstream in(path);
    return in.good();
}

static double now_s()
{
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double>(clock::now().time_since_epoch()).count();
}

__global__ void gated_empty_kernel(unsigned int *sink)
{
    if (threadIdx.x == 0)
        sink[0] = 0x5eed1234u;
}

static const char *driver_ptx = R"ptx(
.version 7.8
.target sm_70
.address_size 64

.visible .entry gated_driver_empty()
{
    ret;
}
)ptx";

static void wait_for_go_file(const std::string &ready_file, const std::string &go_file)
{
    if (!ready_file.empty()) {
        std::ofstream ready(ready_file);
        ready << "ready\n";
    }

    if (go_file.empty())
        return;

    while (!file_exists(go_file))
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
}

static void write_marker_file(const std::string &path, const char *contents)
{
    if (path.empty())
        return;

    std::ofstream out(path);
    out << contents << "\n";
}

static void wait_for_file_if_set(const std::string &path)
{
    if (path.empty())
        return;

    while (!file_exists(path))
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
}

static void call_optional_hook(const char *name)
{
    void *sym = dlsym(RTLD_DEFAULT, name);
    if (!sym)
        return;

    using hook_fn = void (*)();
    reinterpret_cast<hook_fn>(sym)();
}

static void build_driver_graph(int k, CUfunction fn, CUgraph *graph, CUgraphExec *exec)
{
    CHECK_CU(cuGraphCreate(graph, 0));

    CUgraphNode previous = nullptr;
    for (int i = 0; i < k; ++i) {
        CUDA_KERNEL_NODE_PARAMS params;
        std::memset(&params, 0, sizeof(params));
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
            CHECK_CU(cuGraphAddKernelNode(&node, *graph, &previous, 1, &params));
        else
            CHECK_CU(cuGraphAddKernelNode(&node, *graph, nullptr, 0, &params));
        previous = node;
    }

    CHECK_CU(cuGraphInstantiateWithFlags(exec, *graph, 0));
}

static void build_runtime_graph(int k, cudaStream_t stream, unsigned int *sink,
                                cudaGraph_t *graph, cudaGraphExec_t *exec)
{
    CHECK_CUDA(cudaStreamSynchronize(stream));
    CHECK_CUDA(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    for (int i = 0; i < k; ++i)
        gated_empty_kernel<<<1, 1, 0, stream>>>(sink);
    CHECK_CUDA(cudaStreamEndCapture(stream, graph));
    CHECK_CUDA(cudaGraphInstantiate(exec, *graph, nullptr, nullptr, 0));
}

int main()
{
    const int n = env_int("LPG_N", 1000);
    const int warmup_n = env_int("LPG_WARMUP_N", 100);
    const int graph_k = env_int("LPG_GRAPH_K", 300);
    const std::string mode = env_string("LPG_MODE", "driver");
    const std::string ready_file = env_string("LPG_READY_FILE", "");
    const std::string go_file = env_string("LPG_GO_FILE", "");
    const std::string done_file = env_string("LPG_DONE_FILE", "");
    const std::string cleanup_file = env_string("LPG_CLEANUP_FILE", "");

    const bool mode_runtime = mode == "runtime" || mode == "runtime_sync";
    const bool mode_driver = mode == "driver" || mode == "driver_sync";
    const bool mode_runtime_graph = mode == "runtime_graph" || mode == "runtime_graph_sync";
    const bool mode_driver_graph = mode == "driver_graph" || mode == "driver_graph_sync";
    const bool mode_sync_each = mode == "runtime_sync" || mode == "driver_sync" ||
                                mode == "runtime_graph_sync" || mode == "driver_graph_sync";
    const bool mode_graph = mode_runtime_graph || mode_driver_graph;
    if (!mode_runtime && !mode_driver && !mode_runtime_graph && !mode_driver_graph) {
        std::cerr << "unknown LPG_MODE=" << mode
                  << " (expected runtime, driver, runtime_graph, driver_graph, "
                     "or *_sync variants)\n";
        return 2;
    }
    if (n <= 0 || warmup_n < 0 || graph_k <= 0) {
        std::cerr << "invalid LPG_N/LPG_WARMUP_N/LPG_GRAPH_K values\n";
        return 2;
    }

    CHECK_CUDA(cudaFree(nullptr));

    cudaStream_t runtime_stream = nullptr;
    CHECK_CUDA(cudaStreamCreateWithFlags(&runtime_stream, cudaStreamNonBlocking));
    unsigned int *runtime_sink = nullptr;
    CHECK_CUDA(cudaMalloc(&runtime_sink, sizeof(*runtime_sink)));
    CHECK_CUDA(cudaMemsetAsync(runtime_sink, 0, sizeof(*runtime_sink), runtime_stream));
    CHECK_CUDA(cudaStreamSynchronize(runtime_stream));

    CUdevice device = 0;
    CUcontext context = nullptr;
    CUmodule module = nullptr;
    CUfunction driver_fn = nullptr;
    CUstream driver_stream = reinterpret_cast<CUstream>(runtime_stream);

    CHECK_CU(cuInit(0));
    CHECK_CU(cuDeviceGet(&device, 0));
    CHECK_CU(cuCtxGetCurrent(&context));
    if (!context) {
        CHECK_CU(cuDevicePrimaryCtxRetain(&context, device));
        CHECK_CU(cuCtxSetCurrent(context));
    }
    CHECK_CU(cuModuleLoadDataEx(&module, driver_ptx, 0, nullptr, nullptr));
    CHECK_CU(cuModuleGetFunction(&driver_fn, module, "gated_driver_empty"));

    cudaGraph_t runtime_graph = nullptr;
    cudaGraphExec_t runtime_graph_exec = nullptr;
    CUgraph driver_graph = nullptr;
    CUgraphExec driver_graph_exec = nullptr;

    if (mode_runtime_graph)
        build_runtime_graph(graph_k, runtime_stream, runtime_sink, &runtime_graph, &runtime_graph_exec);
    if (mode_driver_graph)
        build_driver_graph(graph_k, driver_fn, &driver_graph, &driver_graph_exec);

    for (int i = 0; i < warmup_n; ++i) {
        if (mode_runtime)
            gated_empty_kernel<<<1, 1, 0, runtime_stream>>>(runtime_sink);
        else if (mode_driver)
            CHECK_CU(cuLaunchKernel(driver_fn, 1, 1, 1, 1, 1, 1, 0, driver_stream, nullptr, nullptr));
        else if (mode_runtime_graph)
            CHECK_CUDA(cudaGraphLaunch(runtime_graph_exec, runtime_stream));
        else
            CHECK_CU(cuGraphLaunch(driver_graph_exec, driver_stream));
        if (mode_sync_each)
            CHECK_CUDA(cudaStreamSynchronize(runtime_stream));
    }
    CHECK_CUDA(cudaStreamSynchronize(runtime_stream));

    wait_for_go_file(ready_file, go_file);

    call_optional_hook("evp_trace_reset");
    double t0 = now_s();
    for (int i = 0; i < n; ++i) {
        if (mode_runtime)
            gated_empty_kernel<<<1, 1, 0, runtime_stream>>>(runtime_sink);
        else if (mode_driver)
            CHECK_CU(cuLaunchKernel(driver_fn, 1, 1, 1, 1, 1, 1, 0, driver_stream, nullptr, nullptr));
        else if (mode_runtime_graph)
            CHECK_CUDA(cudaGraphLaunch(runtime_graph_exec, runtime_stream));
        else
            CHECK_CU(cuGraphLaunch(driver_graph_exec, driver_stream));
        if (mode_sync_each)
            CHECK_CUDA(cudaStreamSynchronize(runtime_stream));
    }
    CHECK_CUDA(cudaStreamSynchronize(runtime_stream));
    double elapsed_us = (now_s() - t0) * 1e6;
    call_optional_hook("evp_trace_dump");

    write_marker_file(done_file, "done");
    wait_for_file_if_set(cleanup_file);

    unsigned int sink_snapshot = 0;
    if (mode_runtime || mode_runtime_graph)
        CHECK_CUDA(cudaMemcpy(&sink_snapshot, runtime_sink, sizeof(sink_snapshot),
                              cudaMemcpyDeviceToHost));

    if (runtime_graph_exec)
        CHECK_CUDA(cudaGraphExecDestroy(runtime_graph_exec));
    if (runtime_graph)
        CHECK_CUDA(cudaGraphDestroy(runtime_graph));
    if (driver_graph_exec)
        CHECK_CU(cuGraphExecDestroy(driver_graph_exec));
    if (driver_graph)
        CHECK_CU(cuGraphDestroy(driver_graph));
    CHECK_CU(cuModuleUnload(module));
    CHECK_CUDA(cudaFree(runtime_sink));
    CHECK_CUDA(cudaStreamDestroy(runtime_stream));

    const int kernels_per_iter = mode_graph ? graph_k : 1;
    const double per_iteration_us = elapsed_us / n;

    std::cout << "{\n"
              << "  \"mode\": \"" << mode << "\",\n"
              << "  \"n\": " << n << ",\n"
              << "  \"warmup_n\": " << warmup_n << ",\n"
              << "  \"kernels_per_iter\": " << kernels_per_iter << ",\n"
              << "  \"sink_snapshot\": " << sink_snapshot << ",\n"
              << "  \"elapsed_us\": " << elapsed_us << ",\n"
              << "  \"per_iteration_us\": " << per_iteration_us << ",\n"
              << "  \"per_launch_us\": " << per_iteration_us << ",\n"
              << "  \"per_kernel_us\": " << elapsed_us / (n * kernels_per_iter) << "\n"
              << "}\n";

    return 0;
}
