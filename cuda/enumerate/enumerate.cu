#include <fmt/core.h>

#include <string>
using namespace fmt;

constexpr auto R = "\033[31m";
constexpr auto G = "\033[32m";
constexpr auto Y = "\033[33m";
constexpr auto B = "\033[34m";
constexpr auto GRAY = "\033[37m";
constexpr auto BOLD = "\033[1m";
constexpr auto NC = "\033[0m";

template <typename T1, typename... TS>
auto with_formatting(const std::string& code, const T1& first, TS... args)
  -> std::string {
    return code + format(format("{}", first), args...) + NC;
}

template <typename... T>
auto bold(T... args) -> std::string {
    return with_formatting(BOLD, args...);
}
template <typename... T>
auto blue(T... args) -> std::string {
    return with_formatting(B, args...);
}

int main(int argc, char** argv) {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    print("Enumerated {} devices\n", bold(deviceCount));
    for (int device = 0; device < deviceCount; ++device) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, device);
        print("{}: {}\n", device, blue(bold(deviceProp.name)));
        print("        Compute Capability: {}\n",
              bold("{}.{}", deviceProp.major, deviceProp.minor));
        print("             Max Grid Size: {}\n",
              bold("{} x {} x {}",
                   deviceProp.maxGridSize[0],
                   deviceProp.maxGridSize[1],
                   deviceProp.maxGridSize[2]));
        print("    Max threads in a block: {}\n", bold(deviceProp.maxThreadsPerBlock));
        print("           ATS Page Access: {}\n",
              deviceProp.pageableMemoryAccessUsesHostPageTables ? "Yes" : "No");
    }
}
