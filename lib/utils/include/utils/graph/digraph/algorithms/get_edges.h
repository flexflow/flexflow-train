#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_DIGRAPH_ALGORITHMS_GET_EDGES_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_DIGRAPH_ALGORITHMS_GET_EDGES_H

#include "utils/graph/digraph/digraph_view.h"

namespace FlexFlow {

size_t num_edges(DiGraphView const &);
std::unordered_set<DirectedEdge> get_edges(DiGraphView const &);

} // namespace FlexFlow

#endif
