#include "compiler/series_parallel/computation_graph/get_computation_graph_series_parallel_decomposition.h"
#include "models/inception_v3/inception_v3.h"
#include "models/split_test/split_test.h"
#include "models/transformer/transformer.h"
#include "utils/benchmark_utils.h"

namespace FlexFlow {

void benchmark_get_computation_graph_series_parallel_decomposition_on_split_test(
    bool dry_run) {
  ComputationGraph cg = get_split_test_computation_graph(/*batch_size=*/8_p);

  LOOP(8, dry_run) {
    get_computation_graph_series_parallel_decomposition(cg);
  }
}

void benchmark_get_computation_graph_series_parallel_decomposition_on_transformer(
    bool dry_run) {
  ComputationGraph cg =
      get_transformer_computation_graph(get_default_transformer_config());

  LOOP(2, dry_run) {
    get_computation_graph_series_parallel_decomposition(cg);
  }
}

void benchmark_get_computation_graph_series_parallel_decomposition_on_inception_v3(
    bool dry_run) {
  ComputationGraph cg = get_inception_v3_computation_graph(
      get_default_inception_v3_training_config());

  LOOP(1, dry_run) {
    get_computation_graph_series_parallel_decomposition(cg);
  }
}

} // namespace FlexFlow
