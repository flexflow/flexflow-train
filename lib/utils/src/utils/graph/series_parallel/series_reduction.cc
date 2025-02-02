#include "utils/graph/series_parallel/series_reduction.h"
#include "utils/containers/contains.h"
#include "utils/containers/contains_key.h"
#include "utils/containers/get_only.h"
#include "utils/containers/require_same.h"
#include "utils/containers/subvec.h"
#include "utils/containers/unordered_set_of.h"
#include "utils/containers/values.h"
#include "utils/graph/digraph/algorithms/get_predecessors.h"
#include "utils/graph/digraph/algorithms/get_topological_ordering.h"
#include "utils/graph/multidigraph/algorithms/get_edges.h"
#include "utils/graph/multidigraph/algorithms/get_incoming_edges.h"
#include "utils/graph/multidigraph/algorithms/get_outgoing_edges.h"
#include "utils/graph/multidigraph/multidiedge.dtg.h"
#include "utils/graph/multidigraph/multidigraph.h"
#include "utils/graph/multidigraph/multidigraph_view.h"
#include "utils/graph/node/algorithms.h"
#include "utils/graph/series_parallel/extended_series_reduction.dtg.h"
#include "utils/hash/unordered_set.h"
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

std::unordered_set<ExtendedSeriesReduction>
    find_all_extended_series_reductions(MultiDiGraphView const &g) {

  auto incoming_edges_map = get_incoming_edges(g, get_nodes(g));
  auto outgoing_edges_map = get_outgoing_edges(g, get_nodes(g));

  std::unordered_map<Node, std::vector<MultiDiEdge>> strands;
  std::unordered_map<Node, Node> node_to_head_of_strand;

  for (Node const &n : get_topological_ordering(g)) {
    if ((incoming_edges_map.at(n).size() == 1) &&
        (outgoing_edges_map.at(n).size() == 1)) {

      MultiDiEdge incoming = get_only(incoming_edges_map.at(n));
      MultiDiEdge outgoing = get_only(outgoing_edges_map.at(n));
      Node pre = g.get_multidiedge_src(incoming);

      if (contains_key(node_to_head_of_strand, pre)) {
        Node head = node_to_head_of_strand.at(pre);
        node_to_head_of_strand.emplace(n, head);
        strands.at(head).push_back(outgoing);

      } else {
        node_to_head_of_strand.emplace(n, n);
        strands[n].push_back(incoming);
        strands[n].push_back(outgoing);
      }
    }
  }

  return transform(unordered_set_of(values(strands)), [&](auto const &edges) {
    return ExtendedSeriesReduction{edges};
  });
}

MultiDiEdge apply_series_reduction(MultiDiGraph &g, SeriesReduction const &r) {
  Node pre_node = get_pre_node(g, r);
  Node center_node = get_center_node(g, r);
  Node post_node = get_post_node(g, r);

  g.remove_node(center_node);
  return g.add_edge(pre_node, post_node);
}

MultiDiEdge
    apply_extended_series_reduction(MultiDiGraph &g,
                                    ExtendedSeriesReduction const &reduction) {

  Node first = g.get_multidiedge_src(reduction.edges.at(0));
  Node last = g.get_multidiedge_dst(reduction.edges.back());

  std::vector<Node> internal_nodes;
  for (MultiDiEdge const &e : subvec(reduction.edges, std::nullopt, -1)) {
    internal_nodes.push_back(g.get_multidiedge_dst(e));
  }

  for (Node const &n : internal_nodes) {
    g.remove_node(n);
  }
  return g.add_edge(first, last);
}
} // namespace FlexFlow
