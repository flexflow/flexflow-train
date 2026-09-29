#include "benchmark/compiler/series_parallel/computation_graph/get_computation_graph_series_parallel_decomposition.h"
#include "utils/benchmark_utils.h"
#include <string>
#include <vector>

using namespace ::FlexFlow;

int main(int argc, char **argv) {
  std::map<std::string, std::function<void(bool)>> benchmarks = {
      {
          "get_computation_graph_series_parallel_decomposition_on_split_test",
          benchmark_get_computation_graph_series_parallel_decomposition_on_split_test,
      },
      {
          "get_computation_graph_series_parallel_decomposition_on_transformer",
          benchmark_get_computation_graph_series_parallel_decomposition_on_transformer,
      },
      {
          "get_computation_graph_series_parallel_decomposition_on_inception_v3",
          benchmark_get_computation_graph_series_parallel_decomposition_on_inception_v3,
      },
  };

  benchmark_main(argc, argv, benchmarks);
}
