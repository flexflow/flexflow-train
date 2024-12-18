#include "utils/graph/serial_parallel/sp_ization/dependencies_are_maintained.h" 
#include "utils/containers/is_subseteq_of.h"
#include "utils/graph/algorithms.h"
#include "utils/graph/digraph/algorithms/get_ancestors.h"
#include "utils/graph/digraph/algorithms/get_descendants.h"
#include "utils/graph/node/algorithms.h"
#include "utils/graph/serial_parallel/digraph_generation.h"
#include "utils/graph/serial_parallel/get_ancestors.h"
#include "utils/graph/serial_parallel/serial_parallel_decomposition.h"

namespace FlexFlow {

bool dependencies_are_maintained(DiGraphView const &g,
                         SerialParallelDecomposition const &sp) {
  assert(has_no_duplicate_nodes(sp));
  if (get_nodes(sp) != get_nodes(g)) {
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
