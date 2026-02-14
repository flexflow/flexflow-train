#include "utils/graph/series_parallel/sp_ization/dependencies_are_maintained.h"
#include "utils/containers/is_subseteq_of.h"
#include "utils/containers/unordered_set_of.h"
#include "utils/graph/digraph/algorithms/get_ancestors.h"
#include "utils/graph/node/algorithms.h"
#include "utils/graph/series_parallel/get_ancestors.h"
#include "utils/graph/series_parallel/series_parallel_decomposition.h"
#include <libassert/assert.hpp>

namespace FlexFlow {

bool dependencies_are_maintained(DiGraphView const &g,
                                 SeriesParallelDecomposition const &sp) {
  ASSERT(has_no_duplicate_nodes(sp));
  if (unordered_set_of(get_nodes(sp)) != get_nodes(g)) {
    return false;
  }

  for (Node const &n : get_nodes(g)) {
    std::unordered_set<Node> ancestors_in_g = get_ancestors(g, n);
    std::unordered_set<Node> ancestors_in_sp = get_ancestors(sp, n);
    if (!is_subseteq_of(ancestors_in_g, ancestors_in_sp)) {
      return false;
    }
  }
  return true;
}

} // namespace FlexFlow
