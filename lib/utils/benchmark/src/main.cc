#include "benchmark/utils/bidict/algorithms/try_merge_nondisjoint_bidicts.h"
#include "benchmark/utils/containers/contains.h"
#include "benchmark/utils/containers/transform.h"
#include "benchmark/utils/containers/try_at.h"
#include "benchmark/utils/containers/try_merge_nondisjoint_maps.h"
#include "benchmark/utils/graph/digraph/algorithms/transitive_closure.h"
#include "benchmark/utils/graph/digraph/algorithms/transitive_reduction.h"
#include "benchmark/utils/graph/open_kwarg_dataflow_graph/algorithms/get_all_open_kwarg_dataflow_edges.h"
#include "benchmark/utils/graph/open_kwarg_dataflow_graph/algorithms/get_open_kwarg_dataflow_subgraph_incoming_edges.h"
#include "benchmark/utils/graph/query_set.h"
#include "benchmark/utils/unordered_bidict/algorithms/try_merge_nondisjoint_unordered_bidicts.h"
#include "utils/benchmark_utils/benchmark_main.h"
#include <map>
#include <string>

using namespace ::FlexFlow;

int main(int argc, char **argv) {
  std::map<std::string, std::function<void(bool)>> benchmarks = {
      {
          "contains_for_set",
          benchmark_contains_for_set,
      },
      {
          "transform_set",
          benchmark_transform_set,
      },
      {
          "try_at_for_map",
          benchmark_try_at_for_map,
      },
      {
          "try_at_for_unordered_map",
          benchmark_try_at_for_unordered_map,
      },
      {
          "try_merge_nondisjoint_maps",
          benchmark_try_merge_nondisjoint_maps,
      },
      {
          "try_merge_nondisjoint_bidicts",
          benchmark_try_merge_nondisjoint_bidicts,
      },
      {
          "try_merge_nondisjoint_unordered_bidicts",
          benchmark_try_merge_nondisjoint_unordered_bidicts,
      },
      {
          "query_set_allowed_values",
          benchmark_query_set_allowed_values,
      },
      {
          "query_set_apply_query_to_set",
          benchmark_query_set_apply_query_to_set,
      },
      {
          "query_set_apply_query_to_vector",
          benchmark_query_set_apply_query_to_vector,
      },
      {
          "transitive_closure",
          benchmark_transitive_closure,
      },
      {
          "transitive_reduction",
          benchmark_transitive_reduction,
      },
      {
          "get_open_kwarg_dataflow_subgraph_incoming_edges_unlabelled",
          benchmark_get_open_kwarg_dataflow_subgraph_incoming_edges_unlabelled,
      },
      {
          "get_open_kwarg_dataflow_subgraph_incoming_edges_labelled",
          benchmark_get_open_kwarg_dataflow_subgraph_incoming_edges_labelled,
      },
      {
          "get_all_open_kwarg_dataflow_edges_unlabelled",
          benchmark_get_all_open_kwarg_dataflow_edges_unlabelled,
      },
      {
          "get_all_open_kwarg_dataflow_edges_labelled",
          benchmark_get_all_open_kwarg_dataflow_edges_labelled,
      },
  };

  benchmark_main(argc, argv, benchmarks);
}
