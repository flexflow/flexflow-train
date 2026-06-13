include(aliasing)
find_package(benchmark REQUIRED)
alias_library(deps::gbenchmark benchmark::benchmark)
alias_library(deps::gbenchmark-main benchmark::benchmark_main)
