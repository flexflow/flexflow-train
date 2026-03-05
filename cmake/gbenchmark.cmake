include(aliasing)

find_package(benchmark REQUIRED)
alias_library(gbenchmark benchmark::benchmark)
alias_library(gbenchmark-main benchmark::benchmark_main)
