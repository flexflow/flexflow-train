#include "compiler/series_parallel/computation_graph/get_computation_graph_series_parallel_decomposition.h"
#include "models/split_test/split_test.h"
#include "models/transformer/transformer.h"
#include "models/inception_v3/inception_v3.h"
#include "models/candle_uno/candle_uno.h"
#include "models/bert/bert.h"
#include <benchmark/benchmark.h>

using namespace ::FlexFlow;

static void benchmark_get_computation_graph_series_parallel_decomposition(benchmark::State &state, ComputationGraph const &cg) {
  // ComputationGraph cg = state.range(0);
      // get_split_test_computation_graph(/*batch_size=*/8);

  for (auto _ : state) {
    get_computation_graph_series_parallel_decomposition(cg);
  }
}

BENCHMARK_CAPTURE(
  benchmark_get_computation_graph_series_parallel_decomposition,
  split_test,
  get_split_test_computation_graph(/*batch_size=*/8)
);

BENCHMARK_CAPTURE(
  benchmark_get_computation_graph_series_parallel_decomposition,
  transformer,
  get_transformer_computation_graph(get_default_transformer_config())
);

BENCHMARK_CAPTURE(
  benchmark_get_computation_graph_series_parallel_decomposition,
  bert,
  get_bert_computation_graph(get_default_bert_config())
);

BENCHMARK_CAPTURE(
  benchmark_get_computation_graph_series_parallel_decomposition,
  candle_uno,
  get_candle_uno_computation_graph(get_default_candle_uno_config())
);

BENCHMARK_CAPTURE(
  benchmark_get_computation_graph_series_parallel_decomposition,
  inception_v3,
  get_inception_v3_computation_graph(get_default_inception_v3_training_config())
);
