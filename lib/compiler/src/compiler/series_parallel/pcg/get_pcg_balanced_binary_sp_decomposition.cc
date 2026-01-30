#include "compiler/series_parallel/pcg/get_pcg_balanced_binary_sp_decomposition.h"
#include "compiler/series_parallel/pcg/get_pcg_series_parallel_decomposition.h"
#include "compiler/series_parallel/pcg/pcg_binary_sp_decomposition.h"
#include "utils/graph/series_parallel/binary_sp_decomposition_tree/balanced_binary_sp_tree_from_nary.h"
#include <optional>

namespace FlexFlow {

std::optional<PCGBinarySPDecomposition>
    get_pcg_balanced_binary_sp_decomposition(
        ParallelComputationGraph const &pcg) {

  std::optional<SeriesParallelDecomposition> spd =
      get_pcg_series_parallel_decomposition(pcg);

  if (!spd.has_value()) {
    return std::nullopt;
  }

  return pcg_binary_sp_decomposition_from_binary_sp_decomposition_tree(
      balanced_binary_sp_tree_from_nary(spd.value()));
}

} // namespace FlexFlow
