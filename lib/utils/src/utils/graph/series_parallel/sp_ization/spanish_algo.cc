#include "utils/graph/series_parallel/sp_ization/spanish_algo.h"
#include "utils/containers/filter_keys.h"
#include "utils/containers/get_only.h"
#include "utils/containers/group_by.h"
#include "utils/containers/intersection.h"
#include "utils/containers/map_values.h"
#include "utils/containers/maximum.h"
#include "utils/containers/range.h"
#include "utils/containers/set_union.h"
#include "utils/containers/transform.h"
#include "utils/containers/values.h"
#include "utils/containers/vector_of.h"
#include "utils/fmt/unordered_multiset.h"
#include "utils/graph/algorithms.h"
#include "utils/graph/digraph/algorithms/get_descendants.h"
#include "utils/graph/digraph/algorithms/get_edges.h"
#include "utils/graph/digraph/algorithms/get_incoming_edges.h"
#include "utils/graph/digraph/algorithms/get_initial_nodes.h"
#include "utils/graph/digraph/algorithms/get_longest_path_lengths_from_root.h"
#include "utils/graph/digraph/algorithms/get_lowest_common_ancestors.h"
#include "utils/graph/digraph/algorithms/get_outgoing_edges.h"
#include "utils/graph/digraph/algorithms/get_successors.h"
#include "utils/graph/digraph/algorithms/get_weakly_connected_components.h"
#include "utils/graph/digraph/algorithms/is_2_terminal_dag.h"
#include "utils/graph/digraph/algorithms/is_acyclic.h"
#include "utils/graph/digraph/algorithms/materialize_digraph_view.h"
#include "utils/graph/digraph/algorithms/transitive_reduction.h"
#include "utils/graph/digraph/digraph.h"
#include "utils/graph/digraph/directed_edge.dtg.h"
#include "utils/graph/instances/adjacency_digraph.h"
#include "utils/graph/node/algorithms.h"
#include "utils/graph/series_parallel/get_series_parallel_decomposition.h"
#include "utils/graph/series_parallel/series_parallel_decomposition.dtg.h"
#include "utils/graph/series_parallel/sp_ization/node_role.dtg.h"
#include "utils/graph/series_parallel/sp_ization/node_role.h"
#include "utils/nonnegative_int/nonnegative_int.h"

#include <unordered_map>
#include <unordered_set>

namespace FlexFlow {

static std::unordered_set<Node>
    filter_sync_nodes(std::unordered_set<Node> const &nodes,
                      std::unordered_map<Node, NodeRole> const &node_roles) {
  return filter(
      nodes, [&](Node const &n) { return node_roles.at(n) != NodeRole::SYNC; });
}

static int get_max_depth(DiGraph const &sp,
                         std::unordered_map<Node, int> const &depth_map) {
  return maximum(values(filter_keys(
      depth_map, [&](Node const &n) { return contains(get_nodes(sp), n); })));
}

DiGraph add_dummy_nodes(DiGraph g,
                        std::unordered_map<Node, NodeRole> &node_roles) {
  std::unordered_map<Node, int> depth_map = map_values(
      get_longest_path_lengths_from_root(g),
      [](nonnegative_int const &i) { return i.unwrap_nonnegative(); });
  for (DirectedEdge const &e : get_edges(g)) {
    Node src = e.src;
    Node dst = e.dst;
    int depth_diff = depth_map.at(dst) - depth_map.at(src);
    if (depth_diff > 1) {
      g.remove_edge(e);
      Node prev_node = src;
      Node intermediate_node = Node{0};
      for (int i : range(1, depth_diff)) {
        intermediate_node = g.add_node();
        node_roles[intermediate_node] = NodeRole::DUMMY;
        g.add_edge(DirectedEdge{prev_node, intermediate_node});
        prev_node = intermediate_node;
      }
      g.add_edge(DirectedEdge{prev_node, dst});
    }
  }
  return g;
}

std::unordered_set<Node>
    get_component(DiGraph const &g,
                  Node const &node,
                  std::unordered_map<Node, int> const &depth_map,
                  std::unordered_map<Node, NodeRole> const &node_roles) {

  int max_depth = get_max_depth(g, depth_map);
  auto is_in_last_2_layers = [&](Node const &n) {
    if (node_roles.at(n) == NodeRole::SYNC) {
      if (get_successors(g, n).empty()) {
        return true;
      }
      int successors_depth =
          get_only(transform(get_successors(g, n),
                             [&](Node const &n) { return depth_map.at(n); }));
      return successors_depth == max_depth;
    } else {
      return (depth_map.at(n) == max_depth) ||
             (depth_map.at(n) == max_depth - 1);
    }
  };
  std::unordered_set<Node> last_two_layers_nodes =
      filter(get_nodes(g), is_in_last_2_layers);

  DiGraph subgraph = materialize_digraph_view<AdjacencyDiGraph>(
      get_subgraph(g, last_two_layers_nodes));
  std::unordered_set<Node> component =
      get_only(filter(get_weakly_connected_components(subgraph),
                      [&](std::unordered_set<Node> const &component) {
                        return contains(component, node);
                      }));
  std::unordered_set<Node> component_without_sync_nodes =
      filter_sync_nodes(component, node_roles);
  return component_without_sync_nodes;
}

static std::unordered_set<Node>
    get_forest_spanish(DiGraph const &g,
                       Node const &handle,
                       std::unordered_set<Node> const &component,
                       std::unordered_map<Node, NodeRole> const &node_roles) {
  std::unordered_set<std::unordered_set<Node>> subtrees =
      transform(get_successors(g, handle), [&](Node const &n) {
        return set_union(get_descendants(g, n), {n});
      });
  auto subtrees_overlapping_with_component =
      filter(subtrees, [&](std::unordered_set<Node> subtree) {
        return intersection(subtree, component).size() > 0;
      });
  std::unordered_set<Node> forest =
      set_union(subtrees_overlapping_with_component);
  forest.insert(handle);
  return filter_sync_nodes(forest, node_roles);
}

static std::pair<std::unordered_set<Node>, std::unordered_set<Node>>
    get_up_and_down(DiGraph const &g,
                    std::unordered_set<Node> const &forest,
                    std::unordered_map<Node, int> const &depth_map) {

  int max_depth = get_max_depth(g, depth_map);
  auto grouped_by_depth =
      group_by(forest, [&](Node const &n) { return depth_map.at(n); });
  return {grouped_by_depth.at_l(max_depth - 1),
          grouped_by_depth.at_l(max_depth)};
}

static std::unordered_set<DirectedEdge>
    edges_to_remove(DiGraph const &g,
                    std::unordered_set<Node> const &up,
                    std::unordered_set<Node> const &down) {
  std::unordered_set<DirectedEdge> to_remove;

  for (Node const &u : up) {
    to_remove = set_union(to_remove, get_outgoing_edges(g, u));
  }
  for (Node const &d : down) {
    to_remove = set_union(to_remove, get_incoming_edges(g, d));
  }

  return to_remove;
}

static std::unordered_set<DirectedEdge>
    edges_to_add_spanish(std::unordered_set<Node> const &up,
                         std::unordered_set<Node> const &down,
                         Node const &sync_node) {
  return set_union(transform(up,
                             [&](Node const &u) {
                               return DirectedEdge{u, sync_node};
                             }),
                   transform(down, [&](Node const &d) {
                     return DirectedEdge{sync_node, d};
                   }));
}

SeriesParallelDecomposition spanish_strata_sync(DiGraph g) {
  assert(is_2_terminal_dag(g));
  assert(is_acyclic(g));

  std::unordered_map<Node, NodeRole> node_roles = get_initial_node_role_map(g);

  g = add_dummy_nodes(g, node_roles);
  std::unordered_map<Node, int> depth_map = map_values(
      get_longest_path_lengths_from_root(g),
      [](nonnegative_int const &i) { return i.unwrap_nonnegative(); });

  DiGraph sp = DiGraph::create<AdjacencyDiGraph>();
  Node root = get_only(get_initial_nodes(g));
  sp.add_node_unsafe(root);
  size_t sync_node_counter = maximum(
      transform(get_nodes(g), [&](Node const &n) { return n.raw_uid; }));
  for (Node const &node : get_bfs_ordering(g, {root})) {
    if (node == root) {
      continue;
    }
    sp.add_node_unsafe(node);
    add_edges(sp, vector_of(get_incoming_edges(g, node)));
    std::unordered_set<Node> component =
        get_component(sp, node, depth_map, node_roles);
    Node handle = get_only(get_lowest_common_ancestors(sp, component).value());
    std::unordered_set<Node> forest =
        get_forest_spanish(sp, handle, component, node_roles);
    auto [up, down] = get_up_and_down(sp, forest, depth_map);

    for (DirectedEdge const &e : edges_to_remove(sp, up, down)) {
      sp.remove_edge(e);
    }

    Node sync_node = Node{++sync_node_counter};
    node_roles[sync_node] = NodeRole::SYNC;
    sp.add_node_unsafe(sync_node);
    for (DirectedEdge const &e : edges_to_add_spanish(up, down, sync_node)) {
      sp.add_edge(e);
    }
  }
  sp = delete_nodes_of_given_role(sp, NodeRole::DUMMY, node_roles);
  sp = transitive_reduction(sp);
  sp = delete_nodes_of_given_role(sp, NodeRole::SYNC, node_roles);
  return get_series_parallel_decomposition(sp).value();
}
} // namespace FlexFlow
