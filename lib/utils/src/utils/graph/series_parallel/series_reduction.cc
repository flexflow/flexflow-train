#include "utils/graph/series_parallel/series_reduction.h"
#include "utils/containers/contains.h"
#include "utils/containers/get_only.h"
#include "utils/containers/require_same.h"
#include "utils/graph/multidigraph/algorithms/get_edges.h"
#include "utils/graph/multidigraph/algorithms/get_incoming_edges.h"
#include "utils/graph/multidigraph/algorithms/get_outgoing_edges.h"
#include "utils/graph/multidigraph/multidigraph.h"
#include "utils/graph/multidigraph/multidigraph_view.h"
#include "utils/graph/node/algorithms.h"
#include <unordered_set>

namespace FlexFlow {

Node get_pre_node(MultiDiGraphView const &g, SeriesReduction const &r) {
  return g.get_multidiedge_src(r.first);
}

Node get_post_node(MultiDiGraphView const &g, SeriesReduction const &r) {
  return g.get_multidiedge_dst(r.second);
}

Node get_center_node(MultiDiGraphView const &g, SeriesReduction const &r) {
  return require_same(g.get_multidiedge_dst(r.first),
                      g.get_multidiedge_src(r.second));
}

SeriesReduction make_series_reduction(MultiDiEdge const &e1,
                                      MultiDiEdge const &e2) {
  return SeriesReduction{e1, e2};
}

std::optional<SeriesReduction>
    find_series_reduction(MultiDiGraphView const &g) {
  for (Node const &node : get_nodes(g)) {
    if (get_incoming_edges(g, node).size() == 1 &&
        get_outgoing_edges(g, node).size() == 1) {
      return make_series_reduction(get_only(get_incoming_edges(g, node)),
                                   get_only(get_outgoing_edges(g, node)));
    }
  }
  return std::nullopt;
}

MultiDiEdge apply_series_reduction(MultiDiGraph &g, SeriesReduction const &r) {
  Node pre_node = get_pre_node(g, r);
  Node center_node = get_center_node(g, r);
  Node post_node = get_post_node(g, r);

  g.remove_node(center_node);
  return g.add_edge(pre_node, post_node);
}

} // namespace FlexFlow
