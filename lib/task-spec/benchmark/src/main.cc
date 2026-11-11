#include "benchmark/task-spec/dynamic_graph/dynamic_open_dataflow_graph.h"
#include "utils/benchmark_utils/benchmark_main.h"
#include <map>
#include <string>

using namespace ::FlexFlow;

int main(int argc, char **argv) {
  std::map<std::string, std::function<void(bool)>> benchmarks = {
      {
          "make_empty_dynamic_open_dataflow_graph",
          benchmark_make_empty_dynamic_open_dataflow_graph,
      },
      {
          "dynamic_open_dataflow_graph_from_invocation_set",
          benchmark_dynamic_open_dataflow_graph_from_invocation_set,
      },
      {
          "dynamic_graph_get_id_for_invocation",
          benchmark_dynamic_graph_get_id_for_invocation,
      },
      {
          "dynamic_graph_get_invocation_for_id",
          benchmark_dynamic_graph_get_invocation_for_id,
      },
      {
          "dynamic_graph_get_id_for_value",
          benchmark_dynamic_graph_get_id_for_value,
      },
      {
          "dynamic_graph_get_value_for_id",
          benchmark_dynamic_graph_get_value_for_id,
      },
  };

  benchmark_main(argc, argv, benchmarks);
}
