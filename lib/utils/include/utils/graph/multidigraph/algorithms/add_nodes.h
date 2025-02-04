#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_MULTIDIGRAPH_ALGORITHMS_ADD_NODES_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_MULTIDIGRAPH_ALGORITHMS_ADD_NODES_H

#include "utils/graph/multidigraph/multidigraph.h"
#include "utils/nonnegative_int/nonnegative_int.h"

namespace FlexFlow {

std::vector<Node> add_nodes(MultiDiGraph &, nonnegative_int num_nodes);

} // namespace FlexFlow

#endif
