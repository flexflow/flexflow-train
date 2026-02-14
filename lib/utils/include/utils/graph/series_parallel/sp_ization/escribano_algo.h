#ifndef _FLEXFLOW_UTILS_GRAPH_SERIES_PARALLEL_SP_IZATION_ESCRIBANO_ALGO_H
#define _FLEXFLOW_UTILS_GRAPH_SERIES_PARALLEL_SP_IZATION_ESCRIBANO_ALGO_H

#include "utils/graph/digraph/digraph.h"
#include "utils/graph/series_parallel/series_parallel_decomposition.dtg.h"
#include "utils/graph/series_parallel/sp_ization/node_role.dtg.h"
#include <unordered_map>
namespace FlexFlow {

DiGraph add_dummy_nodes(DiGraph g,
                        std::unordered_map<Node, NodeRole> &node_roles);

std::unordered_set<Node>
    get_component(DiGraph const &g,
                  Node const &node,
                  std::unordered_map<Node, int> const &depth_map,
                  std::unordered_map<Node, NodeRole> const &node_roles);

SeriesParallelDecomposition escribano_strata_sync(DiGraph g);

} // namespace FlexFlow

#endif

