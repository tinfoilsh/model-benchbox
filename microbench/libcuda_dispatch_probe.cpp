// Inspect libcuda export dispatch stubs in a live process.
//
// NVIDIA's public libcuda entrypoints are small stubs that call through
// writable dispatch slots. This probe finds the indirect call in selected
// stubs, reads the slot target before and after cuInit, and reports offsets
// relative to the loaded libcuda image.

#include <dlfcn.h>

#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

using CUresult = int;
using CuInitFn = CUresult (*)(unsigned int);

struct Probe {
    const char *name;
    void *symbol;
    void *slot;
    void *target;
    int call_offset;
};

static std::uintptr_t ptr_value(const void *ptr)
{
    return reinterpret_cast<std::uintptr_t>(ptr);
}

static std::string dl_path(void *addr, std::uintptr_t *base_out)
{
    Dl_info info;
    std::memset(&info, 0, sizeof(info));
    if (!dladdr(addr, &info))
        return "";

    if (base_out)
        *base_out = ptr_value(info.dli_fbase);
    return info.dli_fname ? info.dli_fname : "";
}

static Probe inspect_symbol(void *handle, const char *name)
{
    Probe probe{name, dlsym(handle, name), nullptr, nullptr, -1};
    if (!probe.symbol)
        return probe;

    auto *bytes = static_cast<unsigned char *>(probe.symbol);
    for (int i = 0; i < 96; ++i) {
        if (bytes[i] == 0xff && bytes[i + 1] == 0x15) {
            std::int32_t disp = 0;
            std::memcpy(&disp, bytes + i + 2, sizeof(disp));
            auto *next = bytes + i + 6;
            probe.slot = next + disp;
            std::memcpy(&probe.target, probe.slot, sizeof(probe.target));
            probe.call_offset = i;
            break;
        }
    }

    return probe;
}

static void print_probe(const Probe &probe)
{
    std::uintptr_t symbol_base = 0;
    std::uintptr_t target_base = 0;
    std::string symbol_path = dl_path(probe.symbol, &symbol_base);
    std::string target_path = dl_path(probe.target, &target_base);

    std::cout << probe.name << "\n";
    std::cout << "  symbol: 0x" << std::hex << ptr_value(probe.symbol);
    if (symbol_base)
        std::cout << " offset 0x" << (ptr_value(probe.symbol) - symbol_base);
    std::cout << std::dec << "\n";
    std::cout << "  symbol_path: " << symbol_path << "\n";
    std::cout << "  call_offset: " << probe.call_offset << "\n";
    std::cout << "  slot: 0x" << std::hex << ptr_value(probe.slot);
    if (symbol_base)
        std::cout << " offset 0x" << (ptr_value(probe.slot) - symbol_base);
    std::cout << std::dec << "\n";
    std::cout << "  target: 0x" << std::hex << ptr_value(probe.target);
    if (target_base)
        std::cout << " offset 0x" << (ptr_value(probe.target) - target_base);
    std::cout << std::dec << "\n";
    std::cout << "  target_path: " << target_path << "\n";
}

int main()
{
    void *handle = dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
    if (!handle) {
        std::cerr << "dlopen libcuda.so.1 failed: " << dlerror() << "\n";
        return 1;
    }

    std::vector<const char *> names = {
        "cuInit",
        "cuLaunchKernel",
        "cuLaunchKernel_ptsz",
        "cuLaunchKernelEx",
        "cuGraphLaunch",
        "cuGraphLaunch_ptsz",
        "cuGraphInstantiateWithFlags",
        "cuGetProcAddress",
        "cuGetProcAddress_v2",
    };

    std::cout << "before cuInit\n";
    for (const char *name : names)
        print_probe(inspect_symbol(handle, name));

    auto cu_init = reinterpret_cast<CuInitFn>(dlsym(handle, "cuInit"));
    if (!cu_init) {
        std::cerr << "dlsym cuInit failed\n";
        return 1;
    }

    CUresult init_result = cu_init(0);
    std::cout << "cuInit_result: " << init_result << "\n";

    std::cout << "after cuInit\n";
    for (const char *name : names)
        print_probe(inspect_symbol(handle, name));

    dlclose(handle);
    return 0;
}
