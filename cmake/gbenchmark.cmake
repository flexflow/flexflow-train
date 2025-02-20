include(aliasing)

if (FF_USE_EXTERNAL_GBENCHMARK)
  find_package(benchmark REQUIRED)
  alias_library(gbenchmark benchmark::benchmark)
  alias_library(gbenchmark-main benchmark::benchmark_main)
else()
  message(FATAL_ERROR "Currently FF_USE_EXTERNAL_GBENCHMARK is required")
endif()

