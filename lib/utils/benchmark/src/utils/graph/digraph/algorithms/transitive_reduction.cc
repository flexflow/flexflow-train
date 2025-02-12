#include "utils/graph/digraph/algorithms/transitive_reduction.h"
#include "./random_dag.h"
#include <benchmark/benchmark.h>

using namespace ::FlexFlow;

static void benchmark_transitive_reduction(benchmark::State &state) {
  int edge_percentage = state.range(0);
  int num_nodes = state.range(1);
  DiGraphView g = random_dag(nonnegative_int{num_nodes},
                             static_cast<float>(edge_percentage) / 100.0);

  for (auto _ : state) {
    transitive_reduction(g);
  }
}

BENCHMARK(benchmark_transitive_reduction)
    ->ArgsProduct({
        benchmark::CreateDenseRange(25, 75, /*step=*/25),
        benchmark::CreateRange(16, 256, /*multi=*/54),
    });
