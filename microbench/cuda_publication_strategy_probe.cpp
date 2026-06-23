// Microbenchmarks for CC host-visible publication strategies.
//
// Build:
//   g++ -O2 -g cuda_publication_strategy_probe.cpp \
//       -o cuda_publication_strategy_probe \
//       -I/usr/local/cuda/include -L/usr/local/cuda/lib64 \
//       -lcudart -lcuda -lpthread -Wl,-rpath,/usr/local/cuda/lib64
//
// Runtime env:
//   CPS_OUT_DIR=/out/publication-strategy
//   CPS_STEPS=64
//   CPS_KERNEL_CYCLES=500000
//   CPS_RING_B=1,4,8,16,32,64,128
//   CPS_RING_K=1,2,4,8,16
//   CPS_RECORD_BYTES=8,16,32,64,128
//   CPS_METADATA_BYTES=4096,65536,262144

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <memory>
#include <mutex>
#include <numeric>
#include <queue>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

double us_between(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double, std::micro>(b - a).count();
}

[[noreturn]] void fail(const std::string &msg) {
    std::fprintf(stderr, "%s\n", msg.c_str());
    std::exit(1);
}

void check_cuda(cudaError_t err, const char *expr, const char *file, int line) {
    if (err != cudaSuccess) {
        std::ostringstream oss;
        oss << "CUDA error at " << file << ":" << line << ": " << expr
            << " returned " << cudaGetErrorString(err) << " (" << int(err)
            << ")";
        fail(oss.str());
    }
}

void check_cu(CUresult err, const char *expr, const char *file, int line) {
    if (err != CUDA_SUCCESS) {
        const char *name = nullptr;
        const char *text = nullptr;
        cuGetErrorName(err, &name);
        cuGetErrorString(err, &text);
        std::ostringstream oss;
        oss << "CU error at " << file << ":" << line << ": " << expr
            << " returned " << (name ? name : "unknown") << ": "
            << (text ? text : "unknown") << " (" << int(err) << ")";
        fail(oss.str());
    }
}

#define CUDA_CHECK(expr) check_cuda((expr), #expr, __FILE__, __LINE__)
#define CU_CHECK(expr) check_cu((expr), #expr, __FILE__, __LINE__)

std::string getenv_str(const char *name, const std::string &def) {
    const char *v = std::getenv(name);
    return v && *v ? std::string(v) : def;
}

int getenv_int(const char *name, int def) {
    const char *v = std::getenv(name);
    return v && *v ? std::atoi(v) : def;
}

unsigned long long getenv_ull(const char *name, unsigned long long def) {
    const char *v = std::getenv(name);
    return v && *v ? std::strtoull(v, nullptr, 10) : def;
}

std::vector<int> parse_int_list(const char *name, const std::vector<int> &def) {
    const char *v = std::getenv(name);
    if (!v || !*v) {
        return def;
    }
    std::vector<int> out;
    std::stringstream ss(v);
    std::string part;
    while (std::getline(ss, part, ',')) {
        if (!part.empty()) {
            out.push_back(std::atoi(part.c_str()));
        }
    }
    return out.empty() ? def : out;
}

void busy_cpu_us(int us) {
    if (us <= 0) {
        return;
    }
    const auto end = Clock::now() + std::chrono::microseconds(us);
    volatile unsigned long long sink = 1;
    while (Clock::now() < end) {
        sink = sink * 1664525ULL + 1013904223ULL;
    }
}

struct DeviceBuffer {
    void *ptr = nullptr;
    size_t bytes = 0;
    DeviceBuffer() = default;
    explicit DeviceBuffer(size_t n) : bytes(n) {
        CUDA_CHECK(cudaMalloc(&ptr, bytes));
    }
    ~DeviceBuffer() {
        if (ptr) {
            cudaFree(ptr);
        }
    }
    DeviceBuffer(const DeviceBuffer &) = delete;
    DeviceBuffer &operator=(const DeviceBuffer &) = delete;
};

struct HostPinnedBuffer {
    void *ptr = nullptr;
    size_t bytes = 0;
    HostPinnedBuffer() = default;
    explicit HostPinnedBuffer(size_t n) : bytes(n) {
        CUDA_CHECK(cudaMallocHost(&ptr, bytes));
        std::memset(ptr, 0, bytes);
    }
    ~HostPinnedBuffer() {
        if (ptr) {
            cudaFreeHost(ptr);
        }
    }
    HostPinnedBuffer(const HostPinnedBuffer &) = delete;
    HostPinnedBuffer &operator=(const HostPinnedBuffer &) = delete;
};

char *cptr(void *p) {
    return reinterpret_cast<char *>(p);
}

const char *kPtx = R"PTX(
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
    mov.u64 %start, %globaltimer;
spin_loop:
    mov.u64 %now, %globaltimer;
    sub.u64 %delta, %now, %start;
    setp.lt.u64 %p, %delta, %cycles;
    @%p bra spin_loop;
    ret;
}

.visible .entry fill_contig_kernel(
    .param .u64 dst,
    .param .u32 words,
    .param .u64 cycles,
    .param .u32 value
)
{
    .reg .pred %p;
    .reg .u64 %dst64, %cycles64, %start, %now, %delta, %idx64, %off64, %addr64;
    .reg .u32 %words32, %value32, %i32, %tmp32;
    ld.param.u64 %dst64, [dst];
    ld.param.u32 %words32, [words];
    ld.param.u64 %cycles64, [cycles];
    ld.param.u32 %value32, [value];
    mov.u64 %start, %globaltimer;
fc_spin:
    mov.u64 %now, %globaltimer;
    sub.u64 %delta, %now, %start;
    setp.lt.u64 %p, %delta, %cycles64;
    @%p bra fc_spin;
    mov.u32 %i32, 0;
fc_store:
    setp.ge.u32 %p, %i32, %words32;
    @%p bra fc_done;
    cvt.u64.u32 %idx64, %i32;
    mul.lo.u64 %off64, %idx64, 4;
    add.u64 %addr64, %dst64, %off64;
    add.u32 %tmp32, %value32, %i32;
    st.global.u32 [%addr64], %tmp32;
    add.u32 %i32, %i32, 1;
    bra fc_store;
fc_done:
    ret;
}

.visible .entry fill_step_major_kernel(
    .param .u64 dst,
    .param .u32 step,
    .param .u32 active_b,
    .param .u32 record_words,
    .param .u64 cycles,
    .param .u32 value
)
{
    .reg .pred %p;
    .reg .u64 %dst64, %cycles64, %start, %now, %delta, %idx64, %off64, %addr64, %base_words64;
    .reg .u32 %step32, %b32, %recw32, %total32, %i32, %value32, %tmp32;
    ld.param.u64 %dst64, [dst];
    ld.param.u32 %step32, [step];
    ld.param.u32 %b32, [active_b];
    ld.param.u32 %recw32, [record_words];
    ld.param.u64 %cycles64, [cycles];
    ld.param.u32 %value32, [value];
    mov.u64 %start, %globaltimer;
fsm_spin:
    mov.u64 %now, %globaltimer;
    sub.u64 %delta, %now, %start;
    setp.lt.u64 %p, %delta, %cycles64;
    @%p bra fsm_spin;
    mul.lo.u32 %total32, %b32, %recw32;
    mul.lo.u32 %tmp32, %step32, %total32;
    cvt.u64.u32 %base_words64, %tmp32;
    mov.u32 %i32, 0;
fsm_store:
    setp.ge.u32 %p, %i32, %total32;
    @%p bra fsm_done;
    cvt.u64.u32 %idx64, %i32;
    add.u64 %idx64, %idx64, %base_words64;
    mul.lo.u64 %off64, %idx64, 4;
    add.u64 %addr64, %dst64, %off64;
    add.u32 %tmp32, %value32, %i32;
    st.global.u32 [%addr64], %tmp32;
    add.u32 %i32, %i32, 1;
    bra fsm_store;
fsm_done:
    ret;
}

.visible .entry fill_request_major_kernel(
    .param .u64 dst,
    .param .u32 step,
    .param .u32 steps,
    .param .u32 active_b,
    .param .u32 record_words,
    .param .u64 cycles,
    .param .u32 value
)
{
    .reg .pred %p;
    .reg .u64 %dst64, %cycles64, %start, %now, %delta, %idx64, %off64, %addr64;
    .reg .u32 %step32, %steps32, %b32, %recw32, %total32, %i32, %r32, %w32, %word_index32, %value32, %tmp32;
    ld.param.u64 %dst64, [dst];
    ld.param.u32 %step32, [step];
    ld.param.u32 %steps32, [steps];
    ld.param.u32 %b32, [active_b];
    ld.param.u32 %recw32, [record_words];
    ld.param.u64 %cycles64, [cycles];
    ld.param.u32 %value32, [value];
    mov.u64 %start, %globaltimer;
frm_spin:
    mov.u64 %now, %globaltimer;
    sub.u64 %delta, %now, %start;
    setp.lt.u64 %p, %delta, %cycles64;
    @%p bra frm_spin;
    mul.lo.u32 %total32, %b32, %recw32;
    mov.u32 %i32, 0;
frm_store:
    setp.ge.u32 %p, %i32, %total32;
    @%p bra frm_done;
    div.u32 %r32, %i32, %recw32;
    rem.u32 %w32, %i32, %recw32;
    mul.lo.u32 %word_index32, %r32, %steps32;
    add.u32 %word_index32, %word_index32, %step32;
    mul.lo.u32 %word_index32, %word_index32, %recw32;
    add.u32 %word_index32, %word_index32, %w32;
    cvt.u64.u32 %idx64, %word_index32;
    mul.lo.u64 %off64, %idx64, 4;
    add.u64 %addr64, %dst64, %off64;
    add.u32 %tmp32, %value32, %i32;
    st.global.u32 [%addr64], %tmp32;
    add.u32 %i32, %i32, 1;
    bra frm_store;
frm_done:
    ret;
}
)PTX";

struct Kernels {
    CUmodule module = nullptr;
    CUfunction spin = nullptr;
    CUfunction fill_contig = nullptr;
    CUfunction fill_step_major = nullptr;
    CUfunction fill_request_major = nullptr;
};

Kernels init_kernels() {
    CUDA_CHECK(cudaFree(nullptr));
    CU_CHECK(cuInit(0));
    CUcontext ctx = nullptr;
    CU_CHECK(cuCtxGetCurrent(&ctx));
    if (!ctx) {
        fail("No current CUDA context");
    }
    Kernels k;
    CU_CHECK(cuModuleLoadData(&k.module, kPtx));
    CU_CHECK(cuModuleGetFunction(&k.spin, k.module, "spin_kernel"));
    CU_CHECK(cuModuleGetFunction(&k.fill_contig, k.module, "fill_contig_kernel"));
    CU_CHECK(cuModuleGetFunction(&k.fill_step_major, k.module, "fill_step_major_kernel"));
    CU_CHECK(cuModuleGetFunction(&k.fill_request_major, k.module, "fill_request_major_kernel"));
    return k;
}

void launch_spin(const Kernels &k, cudaStream_t stream, unsigned long long cycles) {
    unsigned long long c = cycles;
    void *args[] = {&c};
    CU_CHECK(cuLaunchKernel(k.spin, 1, 1, 1, 1, 1, 1, 0,
                            reinterpret_cast<CUstream>(stream), args, nullptr));
}

void launch_fill_contig(const Kernels &k, cudaStream_t stream, void *dst,
                        unsigned words, unsigned long long cycles, unsigned value) {
    unsigned long long p = reinterpret_cast<unsigned long long>(dst);
    unsigned w = words;
    unsigned long long c = cycles;
    unsigned v = value;
    void *args[] = {&p, &w, &c, &v};
    CU_CHECK(cuLaunchKernel(k.fill_contig, 1, 1, 1, 1, 1, 1, 0,
                            reinterpret_cast<CUstream>(stream), args, nullptr));
}

void launch_fill_step_major(const Kernels &k, cudaStream_t stream, void *dst,
                            unsigned step, unsigned active_b,
                            unsigned record_words, unsigned long long cycles,
                            unsigned value) {
    unsigned long long p = reinterpret_cast<unsigned long long>(dst);
    unsigned s = step, b = active_b, rw = record_words;
    unsigned long long c = cycles;
    unsigned v = value;
    void *args[] = {&p, &s, &b, &rw, &c, &v};
    CU_CHECK(cuLaunchKernel(k.fill_step_major, 1, 1, 1, 1, 1, 1, 0,
                            reinterpret_cast<CUstream>(stream), args, nullptr));
}

void launch_fill_request_major(const Kernels &k, cudaStream_t stream, void *dst,
                               unsigned step, unsigned steps, unsigned active_b,
                               unsigned record_words, unsigned long long cycles,
                               unsigned value) {
    unsigned long long p = reinterpret_cast<unsigned long long>(dst);
    unsigned s = step, n = steps, b = active_b, rw = record_words;
    unsigned long long c = cycles;
    unsigned v = value;
    void *args[] = {&p, &s, &n, &b, &rw, &c, &v};
    CU_CHECK(cuLaunchKernel(k.fill_request_major, 1, 1, 1, 1, 1, 1, 0,
                            reinterpret_cast<CUstream>(stream), args, nullptr));
}

struct WorkerItem {
    int id = 0;
    cudaEvent_t event = nullptr;
    void *src = nullptr;
    void *dst = nullptr;
    size_t bytes = 0;
    std::atomic<bool> done{false};
    double event_wait_us = 0.0;
    double copy_api_us = 0.0;
    double sync_us = 0.0;
};

struct WorkerQueue {
    std::mutex mu;
    std::condition_variable cv;
    std::condition_variable done_cv;
    std::queue<WorkerItem *> q;
    bool stop = false;
    int depth = 0;
    int max_depth = 0;
    std::atomic<bool> failed{false};
};

void publication_worker(WorkerQueue *q) {
    cudaSetDevice(0);
    cudaStream_t stream = nullptr;
    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    while (true) {
        WorkerItem *item = nullptr;
        {
            std::unique_lock<std::mutex> lock(q->mu);
            q->cv.wait(lock, [&] { return q->stop || !q->q.empty(); });
            if (q->q.empty() && q->stop) {
                break;
            }
            item = q->q.front();
            q->q.pop();
            q->depth--;
        }

        auto t0 = Clock::now();
        cudaError_t err = cudaEventSynchronize(item->event);
        auto t1 = Clock::now();
        if (err != cudaSuccess) {
            q->failed.store(true);
        }
        err = cudaMemcpyAsync(item->dst, item->src, item->bytes,
                              cudaMemcpyDeviceToHost, stream);
        auto t2 = Clock::now();
        if (err != cudaSuccess) {
            q->failed.store(true);
        }
        err = cudaStreamSynchronize(stream);
        auto t3 = Clock::now();
        if (err != cudaSuccess) {
            q->failed.store(true);
        }

        item->event_wait_us = us_between(t0, t1);
        item->copy_api_us = us_between(t1, t2);
        item->sync_us = us_between(t2, t3);
        item->done.store(true);
        q->done_cv.notify_all();
    }
    cudaStreamDestroy(stream);
}

void enqueue_item(WorkerQueue &q, WorkerItem *item) {
    {
        std::lock_guard<std::mutex> lock(q.mu);
        q.q.push(item);
        q.depth++;
        q.max_depth = std::max(q.max_depth, q.depth);
    }
    q.cv.notify_one();
}

double wait_item(WorkerQueue &q, WorkerItem *item) {
    auto t0 = Clock::now();
    std::unique_lock<std::mutex> lock(q.mu);
    q.done_cv.wait(lock, [&] { return item->done.load(); });
    auto t1 = Clock::now();
    return us_between(t0, t1);
}

struct WorkerRow {
    std::string variant;
    int cpu_work_us = 0;
    int lag = -1;
    int chunk_k = 1;
    int steps = 0;
    int record_bytes = 0;
    unsigned long long kernel_cycles = 0;
    int items = 0;
    int boundary_calls = 0;
    int max_backlog = 0;
    double scheduler_launch_api_us = 0.0;
    double scheduler_wait_us = 0.0;
    double scheduler_copy_api_us = 0.0;
    double scheduler_sync_us = 0.0;
    double worker_event_wait_us = 0.0;
    double worker_copy_api_us = 0.0;
    double worker_sync_us = 0.0;
    double scheduler_issue_us = 0.0;
    double total_wall_us = 0.0;
};

WorkerRow run_inline_publication(const Kernels &kernels, int steps,
                                 int record_bytes,
                                 unsigned long long cycles,
                                 int cpu_work_us) {
    const size_t total_bytes = size_t(steps) * record_bytes;
    const unsigned words = std::max(1, record_bytes / 4);
    DeviceBuffer dev(total_bytes);
    HostPinnedBuffer host(total_bytes);
    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    WorkerRow row;
    row.variant = "inline_d2h_every_step";
    row.cpu_work_us = cpu_work_us;
    row.steps = steps;
    row.record_bytes = record_bytes;
    row.kernel_cycles = cycles;
    row.boundary_calls = steps;

    auto all0 = Clock::now();
    for (int step = 0; step < steps; ++step) {
        auto t0 = Clock::now();
        launch_fill_contig(kernels, stream, cptr(dev.ptr) + size_t(step) * record_bytes,
                           words, cycles, 0x10000000u + unsigned(step));
        auto t1 = Clock::now();
        row.scheduler_launch_api_us += us_between(t0, t1);

        busy_cpu_us(cpu_work_us);

        auto c0 = Clock::now();
        CUDA_CHECK(cudaMemcpyAsync(cptr(host.ptr) + size_t(step) * record_bytes,
                                   cptr(dev.ptr) + size_t(step) * record_bytes,
                                   record_bytes, cudaMemcpyDeviceToHost, stream));
        auto c1 = Clock::now();
        CUDA_CHECK(cudaStreamSynchronize(stream));
        auto c2 = Clock::now();
        row.scheduler_copy_api_us += us_between(c0, c1);
        row.scheduler_sync_us += us_between(c1, c2);
    }
    auto all1 = Clock::now();
    row.scheduler_issue_us = us_between(all0, all1);
    row.total_wall_us = us_between(all0, all1);
    CUDA_CHECK(cudaStreamDestroy(stream));
    return row;
}

WorkerRow run_worker_publication(const Kernels &kernels, const std::string &variant,
                                 int steps, int record_bytes,
                                 unsigned long long cycles, int lag,
                                 int cpu_work_us, int chunk_k) {
    const size_t total_bytes = size_t(steps) * record_bytes;
    const unsigned words = std::max(1, record_bytes / 4);
    DeviceBuffer dev(total_bytes);
    HostPinnedBuffer host(total_bytes);
    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    WorkerQueue q;
    std::thread worker(publication_worker, &q);
    std::vector<std::unique_ptr<WorkerItem>> items;
    std::vector<WorkerItem *> consume_order;

    WorkerRow row;
    row.variant = variant;
    row.cpu_work_us = cpu_work_us;
    row.lag = lag;
    row.chunk_k = chunk_k;
    row.steps = steps;
    row.record_bytes = record_bytes;
    row.kernel_cycles = cycles;

    auto all0 = Clock::now();
    for (int step = 0; step < steps; ++step) {
        auto t0 = Clock::now();
        launch_fill_contig(kernels, stream, cptr(dev.ptr) + size_t(step) * record_bytes,
                           words, cycles, 0x20000000u + unsigned(step));
        auto t1 = Clock::now();
        row.scheduler_launch_api_us += us_between(t0, t1);

        const bool flush = (chunk_k <= 1) || ((step + 1) % chunk_k == 0) ||
                           (step == steps - 1);
        if (flush) {
            const int chunk_len = chunk_k <= 1 ? 1 : ((step % chunk_k) + 1);
            const int start_step = step - chunk_len + 1;
            auto item = std::make_unique<WorkerItem>();
            item->id = int(items.size());
            item->src = cptr(dev.ptr) + size_t(start_step) * record_bytes;
            item->dst = cptr(host.ptr) + size_t(start_step) * record_bytes;
            item->bytes = size_t(chunk_len) * record_bytes;
            CUDA_CHECK(cudaEventCreateWithFlags(&item->event, cudaEventDisableTiming));

            auto e0 = Clock::now();
            CUDA_CHECK(cudaEventRecord(item->event, stream));
            auto e1 = Clock::now();
            row.scheduler_launch_api_us += us_between(e0, e1);

            WorkerItem *raw = item.get();
            items.push_back(std::move(item));
            consume_order.push_back(raw);
            enqueue_item(q, raw);
            row.boundary_calls++;
        }

        busy_cpu_us(cpu_work_us);

        if (chunk_k <= 1 && lag >= 0 && step - lag >= 0) {
            row.scheduler_wait_us += wait_item(q, consume_order[size_t(step - lag)]);
        }
    }

    auto issue1 = Clock::now();
    row.scheduler_issue_us = us_between(all0, issue1);

    if (chunk_k > 1 && lag >= 0) {
        for (size_t i = 0; i < consume_order.size(); ++i) {
            if (int(i) - lag >= 0) {
                row.scheduler_wait_us += wait_item(q, consume_order[i - size_t(lag)]);
            }
        }
    }
    for (WorkerItem *item : consume_order) {
        if (!item->done.load()) {
            row.scheduler_wait_us += wait_item(q, item);
        }
    }
    auto all1 = Clock::now();

    {
        std::lock_guard<std::mutex> lock(q.mu);
        q.stop = true;
    }
    q.cv.notify_one();
    worker.join();

    row.items = int(items.size());
    row.max_backlog = q.max_depth;
    row.total_wall_us = us_between(all0, all1);
    for (const auto &item : items) {
        row.worker_event_wait_us += item->event_wait_us;
        row.worker_copy_api_us += item->copy_api_us;
        row.worker_sync_us += item->sync_us;
        CUDA_CHECK(cudaEventDestroy(item->event));
    }
    CUDA_CHECK(cudaStreamDestroy(stream));
    if (q.failed.load()) {
        fail("worker thread saw a CUDA error");
    }
    return row;
}

void write_worker_rows(const std::string &path, const std::vector<WorkerRow> &rows) {
    std::ofstream out(path);
    out << "variant,cpu_work_us,lag,chunk_k,steps,record_bytes,kernel_cycles,"
        << "items,boundary_calls,max_backlog,scheduler_launch_api_us,"
        << "scheduler_wait_us,scheduler_copy_api_us,scheduler_sync_us,"
        << "worker_event_wait_us,worker_copy_api_us,worker_sync_us,"
        << "scheduler_issue_us,total_wall_us\n";
    out << std::fixed << std::setprecision(3);
    for (const auto &r : rows) {
        out << r.variant << "," << r.cpu_work_us << "," << r.lag << ","
            << r.chunk_k << "," << r.steps << "," << r.record_bytes << ","
            << r.kernel_cycles << "," << r.items << "," << r.boundary_calls
            << "," << r.max_backlog << "," << r.scheduler_launch_api_us << ","
            << r.scheduler_wait_us << "," << r.scheduler_copy_api_us << ","
            << r.scheduler_sync_us << "," << r.worker_event_wait_us << ","
            << r.worker_copy_api_us << "," << r.worker_sync_us << ","
            << r.scheduler_issue_us << "," << r.total_wall_us << "\n";
    }
}

struct RingRow {
    std::string variant;
    int active_b = 0;
    int chunk_k = 0;
    int record_bytes = 0;
    int steps = 0;
    unsigned long long kernel_cycles = 0;
    int boundary_calls = 0;
    size_t copied_bytes = 0;
    size_t max_slab_bytes = 0;
    double boundary_api_us = 0.0;
    double sync_us = 0.0;
    double launch_api_us = 0.0;
    double total_wall_us = 0.0;
    double p50_latency_steps = 0.0;
    double p99_latency_steps = 0.0;
};

std::pair<double, double> latency_stats(int k) {
    std::vector<int> lats;
    for (int i = 0; i < std::max(1, k); ++i) {
        lats.push_back(std::max(0, k - 1 - i));
    }
    std::sort(lats.begin(), lats.end());
    const auto pct = [&](double p) -> double {
        if (lats.empty()) {
            return 0.0;
        }
        size_t idx = size_t(std::min<double>(lats.size() - 1, std::ceil(p * lats.size()) - 1));
        return double(lats[idx]);
    };
    return {pct(0.50), pct(0.99)};
}

RingRow run_global_chunk(const Kernels &kernels, int steps, int active_b,
                         int chunk_k, int record_bytes,
                         unsigned long long cycles, const std::string &variant) {
    const size_t total_bytes = size_t(steps) * active_b * record_bytes;
    const unsigned record_words = std::max(1, record_bytes / 4);
    DeviceBuffer dev(total_bytes);
    HostPinnedBuffer host(total_bytes);
    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    RingRow row;
    row.variant = variant;
    row.active_b = active_b;
    row.chunk_k = chunk_k;
    row.record_bytes = record_bytes;
    row.steps = steps;
    row.kernel_cycles = cycles;
    auto lat = latency_stats(chunk_k);
    row.p50_latency_steps = lat.first;
    row.p99_latency_steps = lat.second;

    auto all0 = Clock::now();
    int chunk_start = 0;
    for (int step = 0; step < steps; ++step) {
        auto l0 = Clock::now();
        launch_fill_step_major(kernels, stream, dev.ptr, unsigned(step),
                               unsigned(active_b), record_words, cycles,
                               0x30000000u + unsigned(step));
        auto l1 = Clock::now();
        row.launch_api_us += us_between(l0, l1);

        const bool flush = ((step + 1) % chunk_k == 0) || (step == steps - 1);
        if (flush) {
            const int chunk_len = step - chunk_start + 1;
            const size_t bytes = size_t(chunk_len) * active_b * record_bytes;
            const size_t offset = size_t(chunk_start) * active_b * record_bytes;
            auto c0 = Clock::now();
            CUDA_CHECK(cudaMemcpyAsync(cptr(host.ptr) + offset, cptr(dev.ptr) + offset,
                                       bytes, cudaMemcpyDeviceToHost, stream));
            auto c1 = Clock::now();
            CUDA_CHECK(cudaStreamSynchronize(stream));
            auto c2 = Clock::now();
            row.boundary_api_us += us_between(c0, c1);
            row.sync_us += us_between(c1, c2);
            row.boundary_calls++;
            row.copied_bytes += bytes;
            row.max_slab_bytes = std::max(row.max_slab_bytes, bytes);
            chunk_start = step + 1;
        }
    }
    auto all1 = Clock::now();
    row.total_wall_us = us_between(all0, all1);
    CUDA_CHECK(cudaStreamDestroy(stream));
    return row;
}

RingRow run_per_request_staggered(const Kernels &kernels, int steps, int active_b,
                                  int chunk_k, int record_bytes,
                                  unsigned long long cycles) {
    const size_t total_bytes = size_t(steps) * active_b * record_bytes;
    const unsigned record_words = std::max(1, record_bytes / 4);
    DeviceBuffer dev(total_bytes);
    HostPinnedBuffer host(total_bytes);
    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    RingRow row;
    row.variant = "per_request_staggered_chunk";
    row.active_b = active_b;
    row.chunk_k = chunk_k;
    row.record_bytes = record_bytes;
    row.steps = steps;
    row.kernel_cycles = cycles;
    auto lat = latency_stats(chunk_k);
    row.p50_latency_steps = lat.first;
    row.p99_latency_steps = lat.second;

    auto all0 = Clock::now();
    for (int step = 0; step < steps; ++step) {
        auto l0 = Clock::now();
        launch_fill_request_major(kernels, stream, dev.ptr, unsigned(step),
                                  unsigned(steps), unsigned(active_b),
                                  record_words, cycles,
                                  0x40000000u + unsigned(step));
        auto l1 = Clock::now();
        row.launch_api_us += us_between(l0, l1);

        bool any = false;
        if (step >= chunk_k - 1) {
            for (int r = 0; r < active_b; ++r) {
                if ((step + r) % chunk_k != chunk_k - 1) {
                    continue;
                }
                const int start = step - chunk_k + 1;
                const size_t src_off =
                    (size_t(r) * steps + size_t(start)) * record_bytes;
                const size_t dst_off =
                    (size_t(r) * steps + size_t(start)) * record_bytes;
                const size_t bytes = size_t(chunk_k) * record_bytes;
                auto c0 = Clock::now();
                CUDA_CHECK(cudaMemcpyAsync(cptr(host.ptr) + dst_off,
                                           cptr(dev.ptr) + src_off, bytes,
                                           cudaMemcpyDeviceToHost, stream));
                auto c1 = Clock::now();
                row.boundary_api_us += us_between(c0, c1);
                row.boundary_calls++;
                row.copied_bytes += bytes;
                row.max_slab_bytes = std::max(row.max_slab_bytes, bytes);
                any = true;
            }
        }
        if (any) {
            auto s0 = Clock::now();
            CUDA_CHECK(cudaStreamSynchronize(stream));
            auto s1 = Clock::now();
            row.sync_us += us_between(s0, s1);
        }
    }
    auto all1 = Clock::now();
    row.total_wall_us = us_between(all0, all1);
    CUDA_CHECK(cudaStreamDestroy(stream));
    return row;
}

void write_ring_rows(const std::string &path, const std::vector<RingRow> &rows) {
    std::ofstream out(path);
    out << "variant,active_b,chunk_k,record_bytes,steps,kernel_cycles,"
        << "boundary_calls,copied_bytes,max_slab_bytes,boundary_api_us,"
        << "sync_us,launch_api_us,total_wall_us,p50_latency_steps,p99_latency_steps\n";
    out << std::fixed << std::setprecision(3);
    for (const auto &r : rows) {
        out << r.variant << "," << r.active_b << "," << r.chunk_k << ","
            << r.record_bytes << "," << r.steps << "," << r.kernel_cycles
            << "," << r.boundary_calls << "," << r.copied_bytes << ","
            << r.max_slab_bytes << "," << r.boundary_api_us << ","
            << r.sync_us << "," << r.launch_api_us << "," << r.total_wall_us
            << "," << r.p50_latency_steps << "," << r.p99_latency_steps << "\n";
    }
}

struct MetadataRow {
    std::string variant;
    int metadata_bytes = 0;
    int steps = 0;
    unsigned long long kernel_cycles = 0;
    int boundary_calls = 0;
    double launch_api_us = 0.0;
    double boundary_api_us = 0.0;
    double event_api_us = 0.0;
    double sync_us = 0.0;
    double cpu_build_us = 0.0;
    double total_wall_us = 0.0;
    unsigned checksum = 0;
};

unsigned checksum_host_words(const void *ptr, size_t bytes) {
    const unsigned *p = reinterpret_cast<const unsigned *>(ptr);
    size_t words = bytes / sizeof(unsigned);
    unsigned x = 0;
    for (size_t i = 0; i < words; i += std::max<size_t>(1, words / 64)) {
        x ^= p[i] + unsigned(i);
    }
    return x;
}

MetadataRow run_metadata_case(const Kernels &kernels, const std::string &variant,
                              int metadata_bytes, int steps,
                              unsigned long long cycles) {
    const unsigned words = std::max(1, metadata_bytes / 4);
    DeviceBuffer meta(metadata_bytes);
    DeviceBuffer scratch(4096);
    HostPinnedBuffer host(metadata_bytes);
    cudaStream_t stream = nullptr;
    cudaStream_t meta_stream = nullptr;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&meta_stream, cudaStreamNonBlocking));

    MetadataRow row;
    row.variant = variant;
    row.metadata_bytes = metadata_bytes;
    row.steps = steps;
    row.kernel_cycles = cycles;

    auto all0 = Clock::now();
    if (variant == "gpu_derived_d2h_plan") {
        for (int step = 0; step < steps; ++step) {
            auto l0 = Clock::now();
            launch_fill_contig(kernels, stream, meta.ptr, words, cycles,
                               0x50000000u + unsigned(step));
            auto l1 = Clock::now();
            row.launch_api_us += us_between(l0, l1);
            auto c0 = Clock::now();
            CUDA_CHECK(cudaMemcpyAsync(host.ptr, meta.ptr, metadata_bytes,
                                       cudaMemcpyDeviceToHost, stream));
            auto c1 = Clock::now();
            CUDA_CHECK(cudaStreamSynchronize(stream));
            auto c2 = Clock::now();
            row.boundary_api_us += us_between(c0, c1);
            row.sync_us += us_between(c1, c2);
            row.boundary_calls++;
            row.checksum ^= checksum_host_words(host.ptr, metadata_bytes);
        }
    } else if (variant == "h2d_after_work_same_stream") {
        for (int step = 0; step < steps; ++step) {
            auto b0 = Clock::now();
            std::memset(host.ptr, step & 0xff, metadata_bytes);
            auto b1 = Clock::now();
            row.cpu_build_us += us_between(b0, b1);
            auto l0 = Clock::now();
            launch_spin(kernels, stream, cycles);
            auto l1 = Clock::now();
            row.launch_api_us += us_between(l0, l1);
            auto c0 = Clock::now();
            CUDA_CHECK(cudaMemcpyAsync(meta.ptr, host.ptr, metadata_bytes,
                                       cudaMemcpyHostToDevice, stream));
            auto c1 = Clock::now();
            CUDA_CHECK(cudaStreamSynchronize(stream));
            auto c2 = Clock::now();
            row.boundary_api_us += us_between(c0, c1);
            row.sync_us += us_between(c1, c2);
            row.boundary_calls++;
            row.checksum ^= unsigned(step) + unsigned(metadata_bytes);
        }
    } else if (variant == "h2d_prestage_independent_stream") {
        std::vector<cudaEvent_t> events;
        events.resize(size_t(steps));
        for (int step = 0; step < steps; ++step) {
            CUDA_CHECK(cudaEventCreateWithFlags(&events[size_t(step)],
                                                cudaEventDisableTiming));
            auto b0 = Clock::now();
            std::memset(host.ptr, (step + 1) & 0xff, metadata_bytes);
            auto b1 = Clock::now();
            row.cpu_build_us += us_between(b0, b1);
            auto c0 = Clock::now();
            CUDA_CHECK(cudaMemcpyAsync(meta.ptr, host.ptr, metadata_bytes,
                                       cudaMemcpyHostToDevice, meta_stream));
            auto c1 = Clock::now();
            CUDA_CHECK(cudaEventRecord(events[size_t(step)], meta_stream));
            auto c2 = Clock::now();
            CUDA_CHECK(cudaStreamWaitEvent(stream, events[size_t(step)], 0));
            auto c3 = Clock::now();
            launch_spin(kernels, stream, cycles);
            auto c4 = Clock::now();
            row.boundary_api_us += us_between(c0, c1);
            row.event_api_us += us_between(c1, c3);
            row.launch_api_us += us_between(c3, c4);
            row.boundary_calls++;
            row.checksum ^= unsigned(step + 1) + unsigned(metadata_bytes);
        }
        auto s0 = Clock::now();
        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaStreamSynchronize(meta_stream));
        auto s1 = Clock::now();
        row.sync_us += us_between(s0, s1);
        for (cudaEvent_t ev : events) {
            CUDA_CHECK(cudaEventDestroy(ev));
        }
    } else if (variant == "cpu_mirror_no_gpu_boundary") {
        for (int step = 0; step < steps; ++step) {
            auto b0 = Clock::now();
            std::memset(host.ptr, step & 0xff, metadata_bytes);
            row.checksum ^= checksum_host_words(host.ptr, metadata_bytes);
            auto b1 = Clock::now();
            row.cpu_build_us += us_between(b0, b1);
            auto l0 = Clock::now();
            launch_spin(kernels, stream, cycles);
            auto l1 = Clock::now();
            row.launch_api_us += us_between(l0, l1);
        }
        auto s0 = Clock::now();
        CUDA_CHECK(cudaStreamSynchronize(stream));
        auto s1 = Clock::now();
        row.sync_us += us_between(s0, s1);
    } else if (variant == "device_resident_metadata") {
        for (int step = 0; step < steps; ++step) {
            auto l0 = Clock::now();
            launch_fill_contig(kernels, stream, meta.ptr, words, cycles,
                               0x60000000u + unsigned(step));
            auto l1 = Clock::now();
            row.launch_api_us += us_between(l0, l1);
        }
        auto c0 = Clock::now();
        CUDA_CHECK(cudaMemcpyAsync(host.ptr, meta.ptr, metadata_bytes,
                                   cudaMemcpyDeviceToHost, stream));
        auto c1 = Clock::now();
        CUDA_CHECK(cudaStreamSynchronize(stream));
        auto c2 = Clock::now();
        row.boundary_api_us += us_between(c0, c1);
        row.sync_us += us_between(c1, c2);
        row.boundary_calls++;
        row.checksum = checksum_host_words(host.ptr, metadata_bytes);
    } else {
        fail("unknown metadata variant: " + variant);
    }
    auto all1 = Clock::now();
    row.total_wall_us = us_between(all0, all1);
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaStreamDestroy(meta_stream));
    return row;
}

void write_metadata_rows(const std::string &path, const std::vector<MetadataRow> &rows) {
    std::ofstream out(path);
    out << "variant,metadata_bytes,steps,kernel_cycles,boundary_calls,"
        << "launch_api_us,boundary_api_us,event_api_us,sync_us,cpu_build_us,"
        << "total_wall_us,checksum\n";
    out << std::fixed << std::setprecision(3);
    for (const auto &r : rows) {
        out << r.variant << "," << r.metadata_bytes << "," << r.steps << ","
            << r.kernel_cycles << "," << r.boundary_calls << ","
            << r.launch_api_us << "," << r.boundary_api_us << ","
            << r.event_api_us << "," << r.sync_us << "," << r.cpu_build_us
            << "," << r.total_wall_us << "," << r.checksum << "\n";
    }
}

} // namespace

int main() {
    CUDA_CHECK(cudaSetDevice(0));
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::fprintf(stderr, "device=%s cc=%d.%d\n", prop.name, prop.major, prop.minor);

    const std::string out_dir = getenv_str("CPS_OUT_DIR", "/out/publication-strategy");
    const int steps = getenv_int("CPS_STEPS", 64);
    const int worker_steps = getenv_int("CPS_WORKER_STEPS", steps);
    const int ring_steps = getenv_int("CPS_RING_STEPS", steps);
    const int metadata_steps = getenv_int("CPS_METADATA_STEPS", steps);
    const unsigned long long cycles = getenv_ull("CPS_KERNEL_CYCLES", 500000ULL);
    const int worker_record_bytes = getenv_int("CPS_WORKER_RECORD_BYTES", 4);
    const std::vector<int> ring_b =
        parse_int_list("CPS_RING_B", {1, 4, 8, 16, 32, 64, 128});
    const std::vector<int> ring_k =
        parse_int_list("CPS_RING_K", {1, 2, 4, 8, 16});
    const std::vector<int> record_bytes =
        parse_int_list("CPS_RECORD_BYTES", {8, 16, 32, 64, 128});
    const std::vector<int> metadata_bytes =
        parse_int_list("CPS_METADATA_BYTES", {4096, 65536, 262144});

    Kernels kernels = init_kernels();

    std::vector<WorkerRow> worker_rows;
    for (int cpu_us : {0, 50, 100, 250, 500, 1000}) {
        worker_rows.push_back(run_inline_publication(
            kernels, worker_steps, worker_record_bytes, cycles, cpu_us));
    }
    worker_rows.push_back(run_worker_publication(
        kernels, "worker_immediate_consume", worker_steps, worker_record_bytes,
        cycles, 0, 0, 1));
    for (int lag : {1, 2, 4, 8}) {
        worker_rows.push_back(run_worker_publication(
            kernels, "worker_delayed_consume", worker_steps, worker_record_bytes,
            cycles, lag, 0, 1));
    }
    for (int cpu_us : {0, 50, 100, 250, 500, 1000}) {
        worker_rows.push_back(run_worker_publication(
            kernels, "worker_lag4_cpu_overlap", worker_steps, worker_record_bytes,
            cycles, 4, cpu_us, 1));
    }
    for (int k : {2, 4, 8, 16}) {
        worker_rows.push_back(run_worker_publication(
            kernels, "worker_chunked_publication", worker_steps,
            worker_record_bytes, cycles, -1, 0, k));
    }
    write_worker_rows(out_dir + "/worker_publication.csv", worker_rows);

    std::vector<RingRow> ring_rows;
    for (int rb : record_bytes) {
        for (int b : ring_b) {
            ring_rows.push_back(run_global_chunk(kernels, ring_steps, b, 1, rb,
                                                 cycles, "per_step_global_flush"));
            for (int k : ring_k) {
                ring_rows.push_back(run_global_chunk(kernels, ring_steps, b, k, rb,
                                                     cycles, "global_decode_step_chunk"));
                ring_rows.push_back(run_per_request_staggered(
                    kernels, ring_steps, b, k, rb, cycles));
            }
        }
    }
    write_ring_rows(out_dir + "/ring_flush.csv", ring_rows);

    std::vector<MetadataRow> metadata_rows;
    const std::vector<std::string> metadata_variants = {
        "gpu_derived_d2h_plan",
        "h2d_after_work_same_stream",
        "h2d_prestage_independent_stream",
        "cpu_mirror_no_gpu_boundary",
        "device_resident_metadata",
    };
    for (int mb : metadata_bytes) {
        for (const auto &variant : metadata_variants) {
            metadata_rows.push_back(run_metadata_case(
                kernels, variant, mb, metadata_steps, cycles));
        }
    }
    write_metadata_rows(out_dir + "/metadata_prestage.csv", metadata_rows);

    CUDA_CHECK(cudaDeviceSynchronize());
    return 0;
}
