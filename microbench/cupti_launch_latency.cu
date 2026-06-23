// CUPTI launch latency probe.
//
// Captures runtime/driver API activity plus kernel queued/submitted/start/end
// timestamps. This splits a CUDA launch into:
//   API duration: host time inside CUDA API
//   queued -> submitted: command buffer waiting before GPU submission
//   submitted -> start: GPU scheduling/front-end latency after submission
//   start -> end: kernel execution

#include <cuda.h>
#include <cuda_runtime.h>
#include <cupti.h>
#include <cupti_activity.h>
#include <cupti_callbacks.h>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <map>
#include <string>
#include <unordered_map>
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

#define CHECK_CUPTI(call)                                                      \
    do {                                                                       \
        CUptiResult _err = (call);                                             \
        if (_err != CUPTI_SUCCESS) {                                           \
            const char *msg = nullptr;                                         \
            cuptiGetResultString(_err, &msg);                                  \
            std::cerr << "CUPTI error " << (msg ? msg : "?")                 \
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

struct ApiRecord {
    uint32_t correlation_id;
    uint32_t cbid;
    uint64_t start;
    uint64_t end;
    std::string domain;
    std::string name;
};

struct KernelRecord {
    uint32_t correlation_id;
    uint64_t queued;
    uint64_t submitted;
    uint64_t start;
    uint64_t end;
    uint64_t completed;
    std::string name;
};

static std::vector<ApiRecord> g_api_records;
static std::vector<KernelRecord> g_kernel_records;
static uint64_t g_dropped_records = 0;

static int env_int(const char *name, int default_value)
{
    const char *value = std::getenv(name);
    if (!value || !*value)
        return default_value;
    return std::atoi(value);
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

static const char *driver_ptx = R"ptx(
.version 7.8
.target sm_70
.address_size 64

.visible .entry driver_empty()
{
    ret;
}
)ptx";

static void CUPTIAPI buffer_requested(uint8_t **buffer, size_t *size, size_t *max_num_records)
{
    constexpr size_t kSize = 16 * 1024 * 1024;
    *size = kSize;
    *buffer = reinterpret_cast<uint8_t *>(std::malloc(kSize));
    *max_num_records = 0;
    if (!*buffer) {
        std::cerr << "failed to allocate CUPTI activity buffer\n";
        std::exit(1);
    }
}

static const char *callback_name(CUpti_CallbackDomain domain, CUpti_CallbackId cbid)
{
    const char *name = nullptr;
    if (cuptiGetCallbackName(domain, cbid, &name) == CUPTI_SUCCESS && name)
        return name;
    return "unknown";
}

static void process_record(CUpti_Activity *record)
{
    switch (record->kind) {
    case CUPTI_ACTIVITY_KIND_RUNTIME: {
        auto *api = reinterpret_cast<CUpti_ActivityAPI *>(record);
        g_api_records.push_back({
            api->correlationId,
            api->cbid,
            api->start,
            api->end,
            "runtime",
            callback_name(CUPTI_CB_DOMAIN_RUNTIME_API, api->cbid),
        });
        break;
    }
    case CUPTI_ACTIVITY_KIND_DRIVER: {
        auto *api = reinterpret_cast<CUpti_ActivityAPI *>(record);
        g_api_records.push_back({
            api->correlationId,
            api->cbid,
            api->start,
            api->end,
            "driver",
            callback_name(CUPTI_CB_DOMAIN_DRIVER_API, api->cbid),
        });
        break;
    }
    case CUPTI_ACTIVITY_KIND_KERNEL: {
        auto *kernel = reinterpret_cast<CUpti_ActivityKernel10 *>(record);
        g_kernel_records.push_back({
            kernel->correlationId,
            kernel->queued,
            kernel->submitted,
            kernel->start,
            kernel->end,
            kernel->completed,
            kernel->name ? kernel->name : "unknown",
        });
        break;
    }
    default:
        break;
    }
}

static void CUPTIAPI buffer_completed(CUcontext ctx,
                                      uint32_t stream_id,
                                      uint8_t *buffer,
                                      size_t size,
                                      size_t valid_size)
{
    CUpti_Activity *record = nullptr;
    while (cuptiActivityGetNextRecord(buffer, valid_size, &record) == CUPTI_SUCCESS)
        process_record(record);

    size_t dropped = 0;
    cuptiActivityGetNumDroppedRecords(ctx, stream_id, &dropped);
    g_dropped_records += dropped;
    std::free(buffer);
}

static void run_empty_pipelined(int n, cudaStream_t stream)
{
    CHECK_CUDA(cudaStreamSynchronize(stream));
    for (int i = 0; i < n; ++i)
        empty_kernel<<<1, 1, 0, stream>>>();
    CHECK_CUDA(cudaStreamSynchronize(stream));
}

static void run_empty_roundtrip(int n, cudaStream_t stream)
{
    CHECK_CUDA(cudaStreamSynchronize(stream));
    for (int i = 0; i < n; ++i) {
        empty_kernel<<<1, 1, 0, stream>>>();
        CHECK_CUDA(cudaStreamSynchronize(stream));
    }
}

static void run_add_pipelined(int n, cudaStream_t stream, float *x, float *y, float *z, int elems)
{
    CHECK_CUDA(cudaStreamSynchronize(stream));
    for (int i = 0; i < n; ++i)
        add_kernel<<<1, 256, 0, stream>>>(x, y, z, elems);
    CHECK_CUDA(cudaStreamSynchronize(stream));
}

static void run_driver_empty_pipelined(int n, CUfunction fn, CUstream stream)
{
    CHECK_CU(cuStreamSynchronize(stream));
    for (int i = 0; i < n; ++i)
        CHECK_CU(cuLaunchKernel(fn, 1, 1, 1, 1, 1, 1, 0, stream, nullptr, nullptr));
    CHECK_CU(cuStreamSynchronize(stream));
}

static void run_driver_empty_roundtrip(int n, CUfunction fn, CUstream stream)
{
    CHECK_CU(cuStreamSynchronize(stream));
    for (int i = 0; i < n; ++i) {
        CHECK_CU(cuLaunchKernel(fn, 1, 1, 1, 1, 1, 1, 0, stream, nullptr, nullptr));
        CHECK_CU(cuStreamSynchronize(stream));
    }
}

static double ns_to_us(uint64_t ns)
{
    return static_cast<double>(ns) / 1000.0;
}

static bool push_delta(std::vector<double> &values, uint64_t a, uint64_t b)
{
    if (a == CUPTI_TIMESTAMP_UNKNOWN || b == CUPTI_TIMESTAMP_UNKNOWN || b < a)
        return false;
    values.push_back(ns_to_us(b - a));
    return true;
}

struct Summary {
    std::vector<double> api_us;
    std::vector<double> api_start_to_queued_us;
    std::vector<double> queued_to_api_end_us;
    std::vector<double> queued_to_submitted_us;
    std::vector<double> submitted_to_start_us;
    std::vector<double> queued_to_start_us;
    std::vector<double> kernel_us;
    std::vector<double> api_start_to_start_us;
    std::vector<double> api_end_to_start_us;
};

static double percentile(std::vector<double> values, double p)
{
    if (values.empty())
        return 0.0;
    std::sort(values.begin(), values.end());
    size_t idx = static_cast<size_t>((values.size() - 1) * p);
    return values[idx];
}

static double mean(const std::vector<double> &values)
{
    if (values.empty())
        return 0.0;
    double total = 0.0;
    for (double v : values)
        total += v;
    return total / values.size();
}

static void print_summary(const std::string &label, const Summary &s)
{
    auto line = [&](const char *metric, const std::vector<double> &values) {
        std::cout << label << "," << metric
                  << "," << values.size()
                  << "," << mean(values)
                  << "," << percentile(values, 0.50)
                  << "," << percentile(values, 0.90)
                  << "," << percentile(values, 0.99)
                  << "\n";
    };
    line("api_us", s.api_us);
    line("api_start_to_queued_us", s.api_start_to_queued_us);
    line("queued_to_api_end_us", s.queued_to_api_end_us);
    line("queued_to_submitted_us", s.queued_to_submitted_us);
    line("submitted_to_start_us", s.submitted_to_start_us);
    line("queued_to_start_us", s.queued_to_start_us);
    line("kernel_us", s.kernel_us);
    line("api_start_to_start_us", s.api_start_to_start_us);
    line("api_end_to_start_us", s.api_end_to_start_us);
}

int main()
{
    int warmup = env_int("CUPTI_LAUNCH_WARMUP", 50);
    int n = env_int("CUPTI_LAUNCH_N", 300);
    int roundtrip_n = env_int("CUPTI_LAUNCH_ROUNDTRIP_N", 80);
    int elems = env_int("CUPTI_LAUNCH_ELEMS", 256);

    CHECK_CUPTI(cuptiActivityRegisterCallbacks(buffer_requested, buffer_completed));
    CHECK_CUPTI(cuptiActivityEnableLatencyTimestamps(1));
    CHECK_CUPTI(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_RUNTIME));
    CHECK_CUPTI(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_DRIVER));
    CHECK_CUPTI(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_KERNEL));

    CHECK_CUDA(cudaFree(0));
    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    float *x = nullptr, *y = nullptr, *z = nullptr;
    CHECK_CUDA(cudaMalloc(&x, elems * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&y, elems * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&z, elems * sizeof(float)));
    CHECK_CUDA(cudaMemsetAsync(x, 1, elems * sizeof(float), stream));
    CHECK_CUDA(cudaMemsetAsync(y, 2, elems * sizeof(float), stream));
    CHECK_CUDA(cudaMemsetAsync(z, 0, elems * sizeof(float), stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));

    run_empty_pipelined(warmup, stream);
    CHECK_CUPTI(cuptiActivityFlushAll(0));

    CUmodule module;
    CUfunction driver_fn;
    CUstream driver_stream;
    CHECK_CU(cuInit(0));
    CHECK_CU(cuModuleLoadData(&module, driver_ptx));
    CHECK_CU(cuModuleGetFunction(&driver_fn, module, "driver_empty"));
    CHECK_CU(cuStreamCreate(&driver_stream, CU_STREAM_NON_BLOCKING));
    run_driver_empty_pipelined(warmup, driver_fn, driver_stream);
    CHECK_CUPTI(cuptiActivityFlushAll(0));

    size_t api_mark = g_api_records.size();
    size_t kernel_mark = g_kernel_records.size();
    run_empty_pipelined(n, stream);
    CHECK_CUPTI(cuptiActivityFlushAll(0));
    size_t api_after_empty_pipe = g_api_records.size();
    size_t kernel_after_empty_pipe = g_kernel_records.size();

    run_empty_roundtrip(roundtrip_n, stream);
    CHECK_CUPTI(cuptiActivityFlushAll(0));
    size_t api_after_empty_roundtrip = g_api_records.size();
    size_t kernel_after_empty_roundtrip = g_kernel_records.size();

    run_add_pipelined(n, stream, x, y, z, elems);
    CHECK_CUPTI(cuptiActivityFlushAll(0));
    size_t api_after_add_pipe = g_api_records.size();
    size_t kernel_after_add_pipe = g_kernel_records.size();

    run_driver_empty_pipelined(n, driver_fn, driver_stream);
    CHECK_CUPTI(cuptiActivityFlushAll(0));
    size_t api_after_driver_pipe = g_api_records.size();
    size_t kernel_after_driver_pipe = g_kernel_records.size();

    run_driver_empty_roundtrip(roundtrip_n, driver_fn, driver_stream);
    CHECK_CUPTI(cuptiActivityFlushAll(0));
    size_t api_after_driver_roundtrip = g_api_records.size();
    size_t kernel_after_driver_roundtrip = g_kernel_records.size();

    std::unordered_map<uint32_t, ApiRecord> api_by_corr;
    for (const auto &api : g_api_records) {
        if (api.name.find("LaunchKernel") != std::string::npos ||
            api.name.find("cuLaunchKernel") != std::string::npos) {
            api_by_corr[api.correlation_id] = api;
        }
    }

    auto summarize = [&](size_t k0, size_t k1) {
        Summary s;
        for (size_t i = k0; i < k1; ++i) {
            const auto &k = g_kernel_records[i];
            auto it = api_by_corr.find(k.correlation_id);
            if (it != api_by_corr.end()) {
                const auto &api = it->second;
                s.api_us.push_back(ns_to_us(api.end - api.start));
                push_delta(s.api_start_to_queued_us, api.start, k.queued);
                push_delta(s.queued_to_api_end_us, k.queued, api.end);
                push_delta(s.api_start_to_start_us, api.start, k.start);
                push_delta(s.api_end_to_start_us, api.end, k.start);
            }
            push_delta(s.queued_to_submitted_us, k.queued, k.submitted);
            push_delta(s.submitted_to_start_us, k.submitted, k.start);
            push_delta(s.queued_to_start_us, k.queued, k.start);
            push_delta(s.kernel_us, k.start, k.end);
        }
        return s;
    };

    std::cout << "label,metric,count,mean_us,p50_us,p90_us,p99_us\n";
    print_summary("empty_pipelined", summarize(kernel_mark, kernel_after_empty_pipe));
    print_summary("empty_roundtrip", summarize(kernel_after_empty_pipe, kernel_after_empty_roundtrip));
    print_summary("add_pipelined", summarize(kernel_after_empty_roundtrip, kernel_after_add_pipe));
    print_summary("driver_empty_pipelined", summarize(kernel_after_add_pipe, kernel_after_driver_pipe));
    print_summary("driver_empty_roundtrip", summarize(kernel_after_driver_pipe, kernel_after_driver_roundtrip));

    std::map<std::string, int> api_counts;
    for (size_t i = api_mark; i < api_after_driver_roundtrip; ++i)
        api_counts[g_api_records[i].domain + ":" + g_api_records[i].name]++;

    std::cerr << "api_records=" << g_api_records.size()
              << " kernel_records=" << g_kernel_records.size()
              << " dropped_records=" << g_dropped_records << "\n";
    for (const auto &kv : api_counts)
        std::cerr << "api_count " << kv.first << " " << kv.second << "\n";

    CHECK_CUDA(cudaFree(z));
    CHECK_CUDA(cudaFree(y));
    CHECK_CUDA(cudaFree(x));
    CHECK_CUDA(cudaStreamDestroy(stream));
    CHECK_CU(cuStreamDestroy(driver_stream));
    CHECK_CU(cuModuleUnload(module));
    return 0;
}
