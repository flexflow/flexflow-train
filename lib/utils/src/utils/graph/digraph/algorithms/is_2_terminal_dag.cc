#include "utils/graph/digraph/algorithms/get_initial_nodes.h"
#include "utils/graph/digraph/algorithms/get_terminal_nodes.h"
#include "utils/graph/digraph/algorithms/is_acyclic.h"

namespace FlexFlow {

bool is_2_terminal_dag(DiGraphView const &g) {
  return (is_acyclic(g) && (get_initial_nodes(g).size() == 1) &&
          get_terminal_nodes(g).size() == 1);
}

} // namespace FlexFlow
