#ifndef _FLEXFLOW_UTILS_GRAPH_SERIES_PARALLEL_SP_IZATION_NODE_ROLE_H
#define _FLEXFLOW_UTILS_GRAPH_SERIES_PARALLEL_SP_IZATION_NODE_ROLE_H

#include "utils/graph/digraph/digraph.h"
#include "utils/graph/digraph/digraph_view.h"
#include "utils/graph/series_parallel/sp_ization/node_role.dtg.h"
#include <unordered_map>

namespace FlexFlow {

std::unordered_map<Node, NodeRole>
    get_initial_node_role_map(DiGraphView const &g);

/**
 * @brief Contracts out nodes of a given role from the graph.
 * 
 * When a node is contracted out, all its predecessors are connected to all 
 * its successors, and then the node is removed.
 */
DiGraph contract_out_nodes_of_given_role(
    DiGraph g,
    NodeRole const &role,
    std::unordered_map<Node, NodeRole> const &node_roles);

} // namespace FlexFlow

#endif
