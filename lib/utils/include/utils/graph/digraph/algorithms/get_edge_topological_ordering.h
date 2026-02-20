#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_DIGRAPH_ALGORITHMS_GET_EDGE_TOPOLOGICAL_ORDERING_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_DIGRAPH_ALGORITHMS_GET_EDGE_TOPOLOGICAL_ORDERING_H

#include "utils/graph/digraph/digraph_view.h"

namespace FlexFlow {

std::vector<DirectedEdge> get_edge_topological_ordering(DiGraphView const &);

} // namespace FlexFlow

#endif
