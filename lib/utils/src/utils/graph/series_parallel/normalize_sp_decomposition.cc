#include "utils/graph/series_parallel/normalize_sp_decomposition.h"
#include "utils/containers/filter.h"
#include "utils/containers/get_only.h"
#include "utils/containers/transform.h"
#include "utils/graph/series_parallel/series_parallel_decomposition.dtg.h"
#include "utils/graph/series_parallel/series_parallel_decomposition.h"
#include "utils/variant.h"
#include <unordered_set>

namespace FlexFlow {

template <typename T>
static auto filter_empty(T const &container) {
  return filter(container, [](auto const &child) {
    return !is_empty(widen<SeriesParallelDecomposition>(child));
  });
}

SeriesParallelDecomposition normalize_sp_decomposition(Node const &node) {
  return SeriesParallelDecomposition(node);
}

SeriesParallelDecomposition
    normalize_sp_decomposition(SeriesSplit const &serial) {
  std::vector<SeriesParallelDecomposition> normalized_children =
      transform(filter_empty(serial.children), [](auto const &child) {
        return normalize_sp_decomposition(
            widen<SeriesParallelDecomposition>(child));
      });

  if (normalized_children.size() == 1) {
    return get_only(normalized_children);
  }
  return series_composition(normalized_children);
}

SeriesParallelDecomposition
    normalize_sp_decomposition(ParallelSplit const &parallel) {
  std::unordered_multiset<SeriesParallelDecomposition> normalized_children =
      transform(filter_empty(parallel.get_children()), [](auto const &child) {
        return normalize_sp_decomposition(
            widen<SeriesParallelDecomposition>(child));
      });

  if (normalized_children.size() == 1) {
    return get_only(normalized_children);
  }
  return parallel_composition(normalized_children);
}

SeriesParallelDecomposition
    normalize_sp_decomposition(SeriesParallelDecomposition const &sp) {
  return sp.visit<SeriesParallelDecomposition>(
      [](auto const &x) { return normalize_sp_decomposition(x); });
}

} // namespace FlexFlow
