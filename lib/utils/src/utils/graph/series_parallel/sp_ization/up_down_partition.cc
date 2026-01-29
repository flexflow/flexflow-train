#include "utils/graph/series_parallel/sp_ization/up_down_partition.h"
#include "utils/containers/filter.h"
#include "utils/graph/algorithms.h"
#include "utils/graph/digraph/algorithms/get_incoming_edges.h"
#include "utils/graph/digraph/algorithms/get_outgoing_edges.h"

namespace FlexFlow {

std::unordered_set<Node> get_up_frontier(DiGraph const &sp,
                                         UpDownPartition const &partition) {
  DiGraphView up_subgraph = get_subgraph(sp, partition.up);
  return filter(partition.up, [&](Node const &node) {
    return get_outgoing_edges(up_subgraph, node).empty();
  });
}

std::unordered_set<Node> get_down_frontier(DiGraph const &sp,
                                           UpDownPartition const &partition) {
  DiGraphView down_subgraph = get_subgraph(sp, partition.down);
  return filter(partition.down, [&](Node const &node) {
    return get_incoming_edges(down_subgraph, node).empty();
  });
}

} // namespace FlexFlow
