#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_BENCHMARK_UTILS_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_BENCHMARK_UTILS_H

#include <cstdint>
#include <functional>
#include <map>
#include <string>

#define LOOP(n, skip) for (volatile uint64_t i = 0; i < (skip ? 0 : n); i++)

namespace FlexFlow {

void benchmark_main(
    int argc,
    char **argv,
    std::map<std::string, std::function<void(bool)>> const &benchmarks);

} // namespace FlexFlow

#endif
