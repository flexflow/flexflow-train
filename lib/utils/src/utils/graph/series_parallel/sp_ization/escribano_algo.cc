#include "utils/graph/series_parallel/sp_ization/escribano_algo.h"
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
#include "utils/graph/digraph/algorithms/transitive_reduction.h"
#include "utils/graph/digraph/digraph.h"
#include "utils/graph/digraph/directed_edge.dtg.h"
#include "utils/graph/instances/adjacency_digraph.h"
#include "utils/graph/node/algorithms.h"
#include "utils/graph/series_parallel/get_series_parallel_decomposition.h"
#include "utils/graph/series_parallel/sp_ization/node_role.h"
#include "utils/nonnegative_int/nonnegative_int.h"
#include "utils/positive_int/positive_int.h"
#include <libassert/assert.hpp>

#include <unordered_map>
#include <unordered_set>

namespace FlexFlow {

static std::unordered_set<Node> filter_out_sync_nodes(
    std::unordered_set<Node> const &nodes,
    std::unordered_map<Node, NodeRole> const &node_roles) {
  return filter(
      nodes, [&](Node const &n) { return node_roles.at(n) != NodeRole::SYNC; });
}

static nonnegative_int
    get_max_depth(DiGraph const &sp,
                  std::unordered_map<Node, nonnegative_int> const &depth_map) {
  return maximum(values(filter_keys(
      depth_map, [&](Node const &n) { return contains(get_nodes(sp), n); })));
}

DiGraph add_dummy_nodes(DiGraph g,
                        std::unordered_map<Node, NodeRole> &node_roles) {
  std::unordered_map<Node, nonnegative_int> depth_map =
      get_longest_path_lengths_from_root(g);

  for (DirectedEdge const &e : get_edges(g)) {
    Node src = e.src;
    Node dst = e.dst;

    int raw_depth_diff = depth_map.at(dst).unwrap_nonnegative() -
                         depth_map.at(src).unwrap_nonnegative();
    ASSERT(raw_depth_diff > 0,
           "Expected edges to strictly increase longest-path depth");
    positive_int depth_diff = positive_int{raw_depth_diff};

    if (depth_diff > 1_p) {
      g.remove_edge(e);

      Node prev_node = src;
      Node intermediate_node = Node{0};

      for (int _ :
           range(1, depth_diff.int_from_positive_int())) {
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
                  std::unordered_map<Node, nonnegative_int> const &depth_map,
                  std::unordered_map<Node, NodeRole> const &node_roles) {

  nonnegative_int max_depth = get_max_depth(g, depth_map);
  auto is_in_last_2_strata = [&](Node const &n) {
    if (node_roles.at(n) == NodeRole::SYNC) {
      if (get_successors(g, n).empty()) {
        return true;
      }
      nonnegative_int successors_depth =
          get_only(transform(get_successors(g, n),
                             [&](Node const &n) { return depth_map.at(n); }));
      return successors_depth == max_depth;
    } else {
      return (depth_map.at(n) == max_depth) ||
             (depth_map.at(n) + 1_n == max_depth);
    }
  };

  std::unordered_set<Node> last_two_layers_nodes =
      filter(get_nodes(g), is_in_last_2_strata);

  DiGraphView subgraph = get_subgraph(g, last_two_layers_nodes);
  std::unordered_set<Node> component =
      get_only(filter(get_weakly_connected_components(subgraph),
                      [&](std::unordered_set<Node> const &component) {
                        return contains(component, node);
                      }));

  std::unordered_set<Node> component_without_sync_nodes =
      filter_out_sync_nodes(component, node_roles);

  return component_without_sync_nodes;
}

static std::unordered_set<Node>
    get_forest_escribano(DiGraph const &g,
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

  return filter_out_sync_nodes(forest, node_roles);
}

static std::pair<std::unordered_set<Node>, std::unordered_set<Node>>
    get_up_and_down_sets(
        DiGraph const &g,
        std::unordered_set<Node> const &forest,
        std::unordered_map<Node, nonnegative_int> const &depth_map) {

  nonnegative_int max_depth = get_max_depth(g, depth_map);

  auto grouped_by_depth =
      group_by(forest, [&](Node const &n) { return depth_map.at(n); });

  return make_pair(grouped_by_depth.at_l(
                       nonnegative_int{max_depth.unwrap_nonnegative() - 1}),
                   grouped_by_depth.at_l(max_depth));
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
    edges_to_add_escribano(std::unordered_set<Node> const &up,
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

static Node add_sync_node(DiGraph &sp,
                          std::unordered_map<Node, NodeRole> &node_roles) {
  Node sync_node = sp.add_node();
  node_roles[sync_node] = NodeRole::SYNC;
  return sync_node;
}

SeriesParallelDecomposition escribano_sp_ization(DiGraph g) {
  ASSERT(is_2_terminal_dag(g));
  ASSERT(is_acyclic(g));

  std::unordered_map<Node, NodeRole> node_roles = get_initial_node_role_map(g);

  g = add_dummy_nodes(g, node_roles);
  std::unordered_map<Node, nonnegative_int> depth_map =
      get_longest_path_lengths_from_root(g);

  DiGraph sp = DiGraph::create<AdjacencyDiGraph>();
  Node root = get_only(get_initial_nodes(g));
  sp.add_node_unsafe(root);

  for (Node const &node : get_bfs_ordering(g, {root})) {
    if (node == root) {
      continue;
    }
    // here we add node unsafe so that we don't have to keep around a mapping
    // between the nodes in g and the nodes in sp. This is safe from having node
    // ids colliding under the assumption that the ids are globally unique.
    sp.add_node_unsafe(node);
    add_edges(sp, get_incoming_edges(g, node));

    std::unordered_set<Node> component =
        get_component(sp, node, depth_map, node_roles);
    Node handle = get_only(get_lowest_common_ancestors(sp, component).value());
    std::unordered_set<Node> forest =
        get_forest_escribano(sp, handle, component, node_roles);
    auto [up, down] = get_up_and_down_sets(sp, forest, depth_map);

    remove_edges(sp, edges_to_remove(sp, up, down));

    Node sync_node = add_sync_node(sp, node_roles);
    add_edges(sp, edges_to_add_escribano(up, down, sync_node));
  }

  // note: for these 3 steps, any order is valid.
  sp = contract_out_nodes_of_given_role(sp, NodeRole::DUMMY, node_roles);
  sp = contract_out_nodes_of_given_role(sp, NodeRole::SYNC, node_roles);
  sp = transitive_reduction(sp);

  return get_series_parallel_decomposition(sp).value();
}
} // namespace FlexFlow
