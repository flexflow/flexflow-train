#include "utils/graph/series_parallel/sp_ization/flexible_algo.h"
#include "utils/containers/all_of.h"
#include "utils/containers/compare_by.h"
#include "utils/containers/contains.h"
#include "utils/containers/filter.h"
#include "utils/containers/generate_map.h"
#include "utils/containers/get_only.h"
#include "utils/containers/intersection.h"
#include "utils/containers/is_subseteq_of.h"
#include "utils/containers/keys.h"
#include "utils/containers/maximum.h"
#include "utils/containers/set_difference.h"
#include "utils/containers/set_union.h"
#include "utils/containers/sorted_by.h"
#include "utils/containers/transform.h"
#include "utils/containers/values.h"
#include "utils/containers/vector_of.h"
#include "utils/graph/algorithms.h"
#include "utils/graph/digraph/algorithms/get_ancestors.h"
#include "utils/graph/digraph/algorithms/get_descendants.h"
#include "utils/graph/digraph/algorithms/get_incoming_edges.h"
#include "utils/graph/digraph/algorithms/get_initial_nodes.h"
#include "utils/graph/digraph/algorithms/get_longest_path_lengths_from_root.h"
#include "utils/graph/digraph/algorithms/get_lowest_common_ancestors.h"
#include "utils/graph/digraph/algorithms/get_outgoing_edges.h"
#include "utils/graph/digraph/algorithms/get_predecessors.h"
#include "utils/graph/digraph/algorithms/get_successors.h"
#include "utils/graph/digraph/algorithms/is_2_terminal_dag.h"
#include "utils/graph/digraph/algorithms/is_acyclic.h"
#include "utils/graph/digraph/algorithms/materialize_digraph_view.h"
#include "utils/graph/digraph/algorithms/transitive_reduction.h"
#include "utils/graph/digraph/digraph.h"
#include "utils/graph/digraph/directed_edge.dtg.h"
#include "utils/graph/instances/adjacency_digraph.h"
#include "utils/graph/node/algorithms.h"
#include "utils/graph/series_parallel/get_series_parallel_decomposition.h"
#include "utils/graph/series_parallel/series_parallel_metrics.h"
#include "utils/graph/series_parallel/sp_ization/dependencies_are_maintained.h"
#include "utils/graph/series_parallel/sp_ization/node_role.h"
#include "utils/graph/series_parallel/sp_ization/up_down_partition.h"
#include <libassert/assert.hpp>

#include <unordered_map>
#include <unordered_set>

namespace FlexFlow {

static std::unordered_set<Node>
    get_component(DiGraph const &sp, std::unordered_set<Node> const &nodes) {
  std::unordered_set<Node> parents = set_union(
      transform(nodes, [&](Node const &n) { return get_predecessors(sp, n); }));
  std::unordered_set<Node> children = set_union(transform(
      parents, [&](Node const &p) { return get_descendants(sp, p); }));
  std::unordered_set<Node> other_parents = set_union(transform(
      children, [&](Node const &c) { return get_predecessors(sp, c); }));
  return set_union(set_union(parents, children), other_parents);
}

static std::unordered_set<Node>
    get_forest_flexible(DiGraph const &sp,
                        Node const &handle,
                        std::unordered_set<Node> const &component,
                        std::unordered_map<Node, NodeRole> const &node_roles) {
  std::unordered_set<std::unordered_set<Node>> subtrees =
      transform(get_successors(sp, handle), [&](Node const &n) {
        return set_union(get_descendants(sp, n), {n});
      });

  std::unordered_set<std::unordered_set<Node>> overlapping_subtrees =
      filter(subtrees, [&](std::unordered_set<Node> const &subtree) {
        return !intersection(subtree, component).empty();
      });

  std::unordered_set<Node> forest = set_union(overlapping_subtrees);
  forest.insert(handle);

  return filter(forest, [&](Node const &n) {
    return node_roles.at(n) != NodeRole::SYNC;
  });
}

static UpDownPartition
    get_up_and_down(DiGraph const &sp,
                    std::unordered_set<Node> const &nodes,
                    std::unordered_set<Node> const &forest,
                    std::unordered_map<Node, float> const &cost_map,
                    std::unordered_map<Node, NodeRole> const &node_roles) {
  DiGraph sp_pure =
      contract_out_nodes_of_given_role(materialize_digraph_view<AdjacencyDiGraph>(sp),
                                       NodeRole::SYNC,
                                       node_roles);

  std::unordered_set<Node> base_down = nodes;
  std::unordered_set<Node> base_up = intersection(
      set_union(transform(
          nodes, [&](Node const &n) { return get_ancestors(sp_pure, n); })),
      forest);
  std::unordered_set<Node> assignable_nodes =
      set_difference(forest, set_union(base_up, base_down));

  DiGraphView forest_subgraph = get_subgraph(sp_pure, forest);
  std::unordered_map<Node, float> critical_path_cost_map =
      get_weighted_longest_path_lengths_from_root(forest_subgraph, cost_map);

  auto get_partition_with_max_up_cost =
      [&](float reference_cost) -> UpDownPartition {
    std::unordered_set<Node> up =
        set_union(base_up, filter(assignable_nodes, [&](Node const &n) {
                    return critical_path_cost_map.at(n) <= reference_cost;
                  }));
    std::unordered_set<Node> down =
        set_difference(set_union(base_down, assignable_nodes), up);
    return UpDownPartition{up, down};
  };

  auto is_valid = [&](UpDownPartition const &partition) -> bool {
    if (!is_subseteq_of(nodes, partition.down)) {
      return false;
    }

    for (Node const &node : get_nodes(sp_pure)) {
      if (contains(partition.down, node)) {
        for (Node const &child : get_successors(sp_pure, node)) {
          if (contains(partition.up, child)) {
            return false;
          }
        }
        for (Node const &parent : get_predecessors(sp_pure, node)) {
          if (contains(forest, parent) && !contains(partition.up, parent) &&
              !contains(partition.down, parent)) {
            return false;
          }
        }
      }
    }
    return true;
  };

  std::unordered_set<UpDownPartition> partitions =
      transform(assignable_nodes, [&](Node const &n) {
        return get_partition_with_max_up_cost(critical_path_cost_map.at(n));
      });
  partitions.insert(
      UpDownPartition{base_up, set_union(base_down, assignable_nodes)});

  std::unordered_set<UpDownPartition> valid_partitions =
      filter(partitions, is_valid);
  ASSERT(!valid_partitions.empty());

  auto partition_cost = [&](UpDownPartition const &p) {
    float up_cost = critical_path_cost(get_subgraph(sp_pure, p.up), cost_map);
    float down_cost =
        critical_path_cost(get_subgraph(sp_pure, p.down), cost_map);
    return std::make_tuple(up_cost + down_cost, down_cost, p.down.size());
  };

  return sorted_by(valid_partitions,
                   compare_by<UpDownPartition>(partition_cost))
      .at(0);
}

static std::unordered_set<DirectedEdge> edges_to_remove_flexible(
    DiGraph const &sp,
    std::unordered_set<Node> const &up,
    std::unordered_set<Node> const &down,
    std::unordered_map<Node, NodeRole> const &node_roles) {
  std::unordered_set<DirectedEdge> to_remove;

  // from up to down
  for (Node const &u : up) {
    for (DirectedEdge const &e : get_outgoing_edges(sp, u)) {
      if (contains(down, e.dst)) {
        to_remove.insert(e);
      }
    }
  }

  for (Node const &node : get_nodes(sp)) {
    if (node_roles.at(node) == NodeRole::SYNC) {
      std::unordered_set<Node> preds = get_predecessors(sp, node);
      std::unordered_set<Node> succs = get_successors(sp, node);
      if (is_subseteq_of(preds, up) && is_subseteq_of(succs, down)) {
        to_remove = set_union(to_remove, get_incoming_edges(sp, node));
        to_remove = set_union(to_remove, get_outgoing_edges(sp, node));
      }
    }
  }

  return to_remove;
}

static std::unordered_set<DirectedEdge>
    edges_to_add_flexible(DiGraph const &sp,
                          UpDownPartition const &partition,
                          Node const &sync_node) {
  std::unordered_set<Node> up_frontier = get_up_frontier(sp, partition);
  std::unordered_set<Node> down_frontier = get_down_frontier(sp, partition);

  return set_union(transform(up_frontier,
                             [&](Node const &u) {
                               return DirectedEdge{u, sync_node};
                             }),
                   transform(down_frontier, [&](Node const &d) {
                     return DirectedEdge{sync_node, d};
                   }));
}

static std::unordered_set<Node>
    get_next_nodes(DiGraph const &sp,
                   DiGraph const &g,
                   std::unordered_map<Node, float> const &cost_map) {
  std::unordered_map<Node, float> sp_longest_paths =
      get_weighted_longest_path_lengths_from_root(sp, cost_map);

  std::unordered_set<Node> sp_nodes = get_nodes(sp);
  std::unordered_set<Node> g_nodes = get_nodes(g);

  // candidate nodes: not in sp but all predecessors in sp
  std::unordered_set<Node> candidate_nodes =
      filter(g_nodes, [&](Node const &node) {
        if (contains(sp_nodes, node)) {
          return false;
        }
        std::unordered_set<Node> preds = get_predecessors(g, node);
        return is_subseteq_of(preds, sp_nodes);
      });

  ASSERT(!candidate_nodes.empty());

  std::unordered_map<Node, float> critical_path_costs =
      generate_map(candidate_nodes, [&](Node const &node) {
        std::unordered_set<Node> preds = get_predecessors(g, node);
        float max_parent_cost = maximum(transform(preds, [&](Node const &pred) {
          return sp_longest_paths.at(pred);
        }));
        return cost_map.at(node) + max_parent_cost;
      });

  Node ref_node =
      sorted_by(candidate_nodes, compare_by<Node>([&](Node const &n) {
                  return std::make_pair(critical_path_costs.at(n), n.raw_uid);
                }))
          .at(0);

  std::unordered_set<Node> ref_preds = get_predecessors(g, ref_node);
  return filter(candidate_nodes, [&](Node const &node) {
    return get_predecessors(g, node) == ref_preds;
  });
}

static bool cost_map_is_valid(DiGraphView const &g,
                              std::unordered_map<Node, float> const &cost_map) {
  bool has_correct_nodes = get_nodes(g) == keys(cost_map);
  bool has_nonnegative_costs =
      all_of(values(cost_map), [&](float const &cost) { return cost >= 0.0f; });
  return has_correct_nodes && has_nonnegative_costs;
}

SeriesParallelDecomposition
    flexible_sync_unchecked(DiGraphView const &g,
                            std::unordered_map<Node, float> cost_map) {
  DiGraph g_reduced =
      materialize_digraph_view<AdjacencyDiGraph>(transitive_reduction(g));

  std::unordered_map<Node, NodeRole> node_roles =
      get_initial_node_role_map(g_reduced);

  DiGraph sp = DiGraph::create<AdjacencyDiGraph>();
  Node root = get_only(get_initial_nodes(g_reduced));
  sp.add_node_unsafe(root);

  while (!is_subseteq_of(get_nodes(g_reduced), get_nodes(sp))) {
    std::unordered_set<Node> nodes = get_next_nodes(sp, g_reduced, cost_map);

    for (Node const &node : nodes) {
      // @colin: not sure if this matches the spec, the counter for the node uid
      // is global and since we have generated these nodes already, we are
      // guaranteed that the uid of the sync nodes will not overlap with them.
      sp.add_node_unsafe(node);
      add_edges(sp, vector_of(get_incoming_edges(g_reduced, node)));
    }
    // TODO(@pietro): ideally optimize this by selectively removing previously
    // added edges
    sp = transitive_reduction(sp);

    std::unordered_set<Node> component = get_component(sp, nodes);
    Node handle = get_only(get_lowest_common_ancestors(sp, component).value());
    std::unordered_set<Node> forest =
        get_forest_flexible(sp, handle, component, node_roles);

    UpDownPartition partition =
        get_up_and_down(sp, nodes, forest, cost_map, node_roles);

    Node sync_node = sp.add_node();
    node_roles[sync_node] = NodeRole::SYNC;
    cost_map[sync_node] = 0.0f;

    for (DirectedEdge const &e : edges_to_remove_flexible(
             sp, partition.up, partition.down, node_roles)) {
      sp.remove_edge(e);
    }
    for (DirectedEdge const &e :
         edges_to_add_flexible(sp, partition, sync_node)) {
      sp.add_edge(e);
    }
  }

  sp = transitive_reduction(sp);
  sp = contract_out_nodes_of_given_role(sp, NodeRole::SYNC, node_roles);

  SeriesParallelDecomposition decomp =
      get_series_parallel_decomposition(sp).value();
  ASSERT(dependencies_are_maintained(g, decomp));

  return decomp;
}

SeriesParallelDecomposition
    flexible_sp_ization(DiGraphView const &g,
                        std::unordered_map<Node, float> const &cost_map) {
  ASSERT(is_2_terminal_dag(g));
  ASSERT(is_acyclic(g));
  ASSERT(cost_map_is_valid(g, cost_map));

  return flexible_sync_unchecked(g, cost_map);
}

} // namespace FlexFlow
