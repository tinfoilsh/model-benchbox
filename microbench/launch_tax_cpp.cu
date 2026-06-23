// Minimal CUDA launch-tax probe for CC command-submission attribution.
//
// This intentionally avoids PyTorch. It measures runtime API launches, direct
// CUDA Driver API launches, graph replay, and launch parameter-size sensitivity.

#include <cuda.h>
#include <cuda_runtime.h>

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

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

static double now_s()
{
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double>(clock::now().time_since_epoch()).count();
}

__global__ void empty_kernel()
{
}

__global__ void add_kernel(float *x, float *y, float *z, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        z[idx] = x[idx] + y[idx];
}

template <int N>
struct Payload {
    int marker;
    unsigned char bytes[N > 0 ? N : 1];
};

template <int N>
__global__ void payload_kernel(Payload<N> payload, int *out)
{
    if (blockIdx.x == 0 && threadIdx.x == 0)
        out[0] = payload.marker + payload.bytes[0] + payload.bytes[N > 1 ? N - 1 : 0];
}

static double measure_runtime_empty_pipelined(int n, cudaStream_t stream)
{
    CHECK_CUDA(cudaStreamSynchronize(stream));
    double t0 = now_s();
    for (int i = 0; i < n; ++i)
        empty_kernel<<<1, 1, 0, stream>>>();
    CHECK_CUDA(cudaStreamSynchronize(stream));
    return (now_s() - t0) / n * 1e6;
}

static double measure_runtime_empty_roundtrip(int n, cudaStream_t stream)
{
    CHECK_CUDA(cudaStreamSynchronize(stream));
    double t0 = now_s();
    for (int i = 0; i < n; ++i) {
        empty_kernel<<<1, 1, 0, stream>>>();
        CHECK_CUDA(cudaStreamSynchronize(stream));
    }
    return (now_s() - t0) / n * 1e6;
}

static double measure_runtime_add_pipelined(int n, cudaStream_t stream, float *x, float *y, float *z, int elems)
{
    CHECK_CUDA(cudaStreamSynchronize(stream));
    double t0 = now_s();
    for (int i = 0; i < n; ++i)
        add_kernel<<<1, 256, 0, stream>>>(x, y, z, elems);
    CHECK_CUDA(cudaStreamSynchronize(stream));
    return (now_s() - t0) / n * 1e6;
}

static double measure_runtime_add_roundtrip(int n, cudaStream_t stream, float *x, float *y, float *z, int elems)
{
    CHECK_CUDA(cudaStreamSynchronize(stream));
    double t0 = now_s();
    for (int i = 0; i < n; ++i) {
        add_kernel<<<1, 256, 0, stream>>>(x, y, z, elems);
        CHECK_CUDA(cudaStreamSynchronize(stream));
    }
    return (now_s() - t0) / n * 1e6;
}

template <int Bytes>
static double measure_payload_pipelined(int n, cudaStream_t stream, int *out)
{
    Payload<Bytes> payload;
    payload.marker = Bytes;
    std::memset(payload.bytes, Bytes & 0xff, sizeof(payload.bytes));

    CHECK_CUDA(cudaStreamSynchronize(stream));
    double t0 = now_s();
    for (int i = 0; i < n; ++i)
        payload_kernel<Bytes><<<1, 1, 0, stream>>>(payload, out);
    CHECK_CUDA(cudaStreamSynchronize(stream));
    return (now_s() - t0) / n * 1e6;
}

static const char *driver_ptx = R"ptx(
.version 7.8
.target sm_70
.address_size 64

.visible .entry driver_empty()
{
    ret;
}
)ptx";

static double measure_driver_empty_pipelined(int n, CUfunction fn, CUstream stream)
{
    CHECK_CU(cuStreamSynchronize(stream));
    double t0 = now_s();
    for (int i = 0; i < n; ++i)
        CHECK_CU(cuLaunchKernel(fn, 1, 1, 1, 1, 1, 1, 0, stream, nullptr, nullptr));
    CHECK_CU(cuStreamSynchronize(stream));
    return (now_s() - t0) / n * 1e6;
}

static double measure_driver_empty_roundtrip(int n, CUfunction fn, CUstream stream)
{
    CHECK_CU(cuStreamSynchronize(stream));
    double t0 = now_s();
    for (int i = 0; i < n; ++i) {
        CHECK_CU(cuLaunchKernel(fn, 1, 1, 1, 1, 1, 1, 0, stream, nullptr, nullptr));
        CHECK_CU(cuStreamSynchronize(stream));
    }
    return (now_s() - t0) / n * 1e6;
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

static double measure_driver_graph_build_us_per_kernel(int k, CUfunction fn)
{
    CUgraph graph = nullptr;
    CUgraphExec exec = nullptr;

    double t0 = now_s();
    build_driver_graph(k, fn, &graph, &exec);
    double us = (now_s() - t0) / k * 1e6;

    CHECK_CU(cuGraphExecDestroy(exec));
    CHECK_CU(cuGraphDestroy(graph));
    return us;
}

static double measure_driver_graph_replay_us(int k, int reps, CUfunction fn, CUstream stream)
{
    CUgraph graph = nullptr;
    CUgraphExec exec = nullptr;
    build_driver_graph(k, fn, &graph, &exec);

    CHECK_CU(cuGraphLaunch(exec, stream));
    CHECK_CU(cuStreamSynchronize(stream));

    double t0 = now_s();
    for (int i = 0; i < reps; ++i)
        CHECK_CU(cuGraphLaunch(exec, stream));
    CHECK_CU(cuStreamSynchronize(stream));
    double us = (now_s() - t0) / reps * 1e6;

    CHECK_CU(cuGraphExecDestroy(exec));
    CHECK_CU(cuGraphDestroy(graph));
    return us;
}

static double measure_graph_capture_us_per_kernel(int k, cudaStream_t stream)
{
    CHECK_CUDA(cudaStreamSynchronize(stream));
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t exec = nullptr;

    double t0 = now_s();
    CHECK_CUDA(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    for (int i = 0; i < k; ++i)
        empty_kernel<<<1, 1, 0, stream>>>();
    CHECK_CUDA(cudaStreamEndCapture(stream, &graph));
    CHECK_CUDA(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
    double dt = now_s() - t0;

    CHECK_CUDA(cudaGraphExecDestroy(exec));
    CHECK_CUDA(cudaGraphDestroy(graph));
    return dt / k * 1e6;
}

static double measure_graph_replay_us(int k, int reps, cudaStream_t stream)
{
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t exec = nullptr;

    CHECK_CUDA(cudaStreamSynchronize(stream));
    CHECK_CUDA(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    for (int i = 0; i < k; ++i)
        empty_kernel<<<1, 1, 0, stream>>>();
    CHECK_CUDA(cudaStreamEndCapture(stream, &graph));
    CHECK_CUDA(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));

    CHECK_CUDA(cudaGraphLaunch(exec, stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));

    double t0 = now_s();
    for (int i = 0; i < reps; ++i)
        CHECK_CUDA(cudaGraphLaunch(exec, stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));
    double us = (now_s() - t0) / reps * 1e6;

    CHECK_CUDA(cudaGraphExecDestroy(exec));
    CHECK_CUDA(cudaGraphDestroy(graph));
    return us;
}

static void json_kv(std::ostream &out, const std::string &key, double value, bool last = false)
{
    out << "  \"" << key << "\": " << value << (last ? "\n" : ",\n");
}

int main()
{
    int n = env_int("LT_N", 2000);
    int roundtrip_n = env_int("LT_ROUNDTRIP_N", 300);
    int warmup_n = env_int("LT_WARMUP_N", 100);
    int graph_k = env_int("LT_GRAPH_K", 300);
    int graph_reps = env_int("LT_GRAPH_REPS", 100);
    int elems = env_int("LT_ELEMS", 256);

    CHECK_CUDA(cudaFree(0));
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));

    cudaStream_t runtime_stream;
    CHECK_CUDA(cudaStreamCreateWithFlags(&runtime_stream, cudaStreamNonBlocking));

    float *x = nullptr, *y = nullptr, *z = nullptr;
    int *out = nullptr;
    CHECK_CUDA(cudaMalloc(&x, elems * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&y, elems * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&z, elems * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&out, sizeof(int)));
    CHECK_CUDA(cudaMemsetAsync(x, 1, elems * sizeof(float), runtime_stream));
    CHECK_CUDA(cudaMemsetAsync(y, 2, elems * sizeof(float), runtime_stream));
    CHECK_CUDA(cudaMemsetAsync(z, 0, elems * sizeof(float), runtime_stream));
    CHECK_CUDA(cudaStreamSynchronize(runtime_stream));

    for (int i = 0; i < 3; ++i)
        (void)measure_runtime_empty_pipelined(warmup_n, runtime_stream);

    CUmodule module;
    CUfunction driver_fn;
    CUstream driver_stream;
    CHECK_CU(cuInit(0));
    CHECK_CU(cuModuleLoadData(&module, driver_ptx));
    CHECK_CU(cuModuleGetFunction(&driver_fn, module, "driver_empty"));
    CHECK_CU(cuStreamCreate(&driver_stream, CU_STREAM_NON_BLOCKING));
    for (int i = 0; i < 3; ++i)
        (void)measure_driver_empty_pipelined(warmup_n, driver_fn, driver_stream);

    std::ostringstream out_json;
    out_json.setf(std::ios::fixed);
    out_json.precision(6);
    out_json << "{\n";
    out_json << "  \"device\": \"" << prop.name << "\",\n";
    out_json << "  \"lt_n\": " << n << ",\n";
    out_json << "  \"lt_roundtrip_n\": " << roundtrip_n << ",\n";
    out_json << "  \"lt_graph_k\": " << graph_k << ",\n";
    out_json << "  \"lt_graph_reps\": " << graph_reps << ",\n";

    json_kv(out_json, "runtime_empty_pipelined_us", measure_runtime_empty_pipelined(n, runtime_stream));
    json_kv(out_json, "runtime_empty_roundtrip_us", measure_runtime_empty_roundtrip(roundtrip_n, runtime_stream));
    json_kv(out_json, "runtime_add_pipelined_us", measure_runtime_add_pipelined(n, runtime_stream, x, y, z, elems));
    json_kv(out_json, "runtime_add_roundtrip_us", measure_runtime_add_roundtrip(roundtrip_n, runtime_stream, x, y, z, elems));
    json_kv(out_json, "payload_0b_pipelined_us", measure_payload_pipelined<0>(n, runtime_stream, out));
    json_kv(out_json, "payload_64b_pipelined_us", measure_payload_pipelined<64>(n, runtime_stream, out));
    json_kv(out_json, "payload_512b_pipelined_us", measure_payload_pipelined<512>(n, runtime_stream, out));
    json_kv(out_json, "payload_2048b_pipelined_us", measure_payload_pipelined<2048>(n, runtime_stream, out));
    json_kv(out_json, "driver_empty_pipelined_us", measure_driver_empty_pipelined(n, driver_fn, driver_stream));
    json_kv(out_json, "driver_empty_roundtrip_us", measure_driver_empty_roundtrip(roundtrip_n, driver_fn, driver_stream));
    json_kv(out_json, "graph_capture_us_per_kernel", measure_graph_capture_us_per_kernel(graph_k, runtime_stream));
    double graph_replay_us = measure_graph_replay_us(graph_k, graph_reps, runtime_stream);
    json_kv(out_json, "graph_replay_us", graph_replay_us);
    json_kv(out_json, "graph_replay_us_per_kernel", graph_replay_us / graph_k);
    json_kv(out_json, "driver_graph_build_us_per_kernel", measure_driver_graph_build_us_per_kernel(graph_k, driver_fn));
    double driver_graph_replay_us = measure_driver_graph_replay_us(graph_k, graph_reps, driver_fn, driver_stream);
    json_kv(out_json, "driver_graph_replay_us", driver_graph_replay_us);
    json_kv(out_json, "driver_graph_replay_us_per_kernel", driver_graph_replay_us / graph_k, true);
    out_json << "}\n";

    std::cout << out_json.str();

    CHECK_CU(cuStreamDestroy(driver_stream));
    CHECK_CU(cuModuleUnload(module));
    CHECK_CUDA(cudaFree(out));
    CHECK_CUDA(cudaFree(z));
    CHECK_CUDA(cudaFree(y));
    CHECK_CUDA(cudaFree(x));
    CHECK_CUDA(cudaStreamDestroy(runtime_stream));
    return 0;
}
