#include "utils/containers/contains.h"
#include "utils/graph/serial_parallel/sp_ization/spanish_algo.h"
#include "utils/containers/filter.h"
#include "utils/containers/get_only.h"
#include "utils/containers/sorted_by.h"
#include "utils/containers/set_difference.h"
#include "utils/containers/transform.h"
#include "utils/graph/algorithms.h"
#include "utils/graph/digraph/algorithms.h"
#include "utils/graph/digraph/algorithms/get_descendants.h"
#include "utils/graph/digraph/algorithms/get_longest_path_lengths_from_root.h"
#include "utils/graph/digraph/algorithms/get_lowest_common_ancestors.h"
#include "utils/graph/digraph/algorithms/get_successors.h"
#include "utils/graph/digraph/algorithms/get_weakly_connected_components.h"
#include "utils/graph/digraph/algorithms/is_2_terminal_dag.h"
#include "utils/graph/instances/adjacency_digraph.h"
#include "utils/graph/node/algorithms.h"
#include "utils/graph/serial_parallel/get_serial_parallel_decomposition.h"
#include "utils/graph/serial_parallel/serial_parallel_decomposition.h"
#include "utils/graph/serial_parallel/sp_ization/is_valid_sp_ization.h"
#include "utils/graph/digraph/algorithms/transitive_reduction.h"
#include "utils/graph/digraph/algorithms/transitive_reduction.h"
#include "utils/graph/digraph/algorithms/materialize_digraph_view.h"

namespace FlexFlow {

static std::unordered_set<Node> find_down_class_which_contains_node(
    DiGraphView const &g,
    Node const &n,
    std::unordered_map<Node, int> const &depth_map) {
  int max_depth = depth_map.at(n);
  std::unordered_set<Node> last_2_layers =
      filter(get_nodes(g), [&](Node const &node) {
        return (depth_map.at(node) == max_depth) ||
               (depth_map.at(node) == max_depth - 1);
      });
  DiGraphView last_2_layers_subgraph = get_subgraph(g, last_2_layers);
  std::unordered_set<Node> component_containing_n =
      get_only(filter(get_weakly_connected_components(last_2_layers_subgraph),
                      [&](std::unordered_set<Node> const &component) {
                        return contains(component, n);
                      }));
  // TODO(@pietro): check that it has height 2
  return set_union(
      transform(get_nodes(last_2_layers_subgraph),
                [&](Node const &node) { return get_successors(g, node); }));
}

static Node get_handle(DiGraphView const &g, std::unordered_set<Node> const &down_class) {
  return get_only(get_lowest_common_ancestors(g, down_class).value());
}


SerialParallelDecomposition
    one_node_at_a_time_spanish_sp_ization_unchecked(DiGraph g) {
  // TODO(@pietro): apply transitive reduction
  g = materialize_digraph_view<AdjacencyDiGraph>(transitive_reduction(g));
  std::unordered_set<Node> original_nodes = get_nodes(g);
  std::unordered_map<Node, int> depth_map =
      get_longest_path_lengths_from_root(g);
  std::vector<Node> nodes =
      sorted_by(get_nodes(g), [&](Node const &n1, Node const &n2) {
        return depth_map.at(n1) < depth_map.at(n2);
      });
  Node root = nodes.at(0);
  std::unordered_set<Node> to_consider = {root};
  for (auto const &n : nodes) {
    std::cout << n << get_edges(g) << std::endl;
    if (n == root) {
      continue;
    }

    to_consider.insert(n);
    DiGraphView subgraph = get_subgraph(g, to_consider);
    std::unordered_set<Node> component =
        find_down_class_which_contains_node(subgraph, n, depth_map);
    
    Node handle = get_handle(subgraph, component);

    std::unordered_set<Node> forest = filter(get_descendants(subgraph, handle),
                    [&](Node const &n) { return contains(original_nodes, n); });
    if (forest.empty()) {continue;}
    std::unordered_set<Node> last_layer = filter(forest, [&](auto const &node) {
      return depth_map.at(node) == depth_map.at(n);
    });
    std::unordered_set<Node> penultimate_layer =
        filter(forest, [&](auto const &node) {
          return depth_map.at(node) == depth_map.at(n) - 1;
        });
    
    std::cout << handle << forest << std::endl << std::endl;
    
    Node sync_node = g.add_node();
    to_consider.insert(sync_node);
    for (DirectedEdge const &e : get_edges(g)) {
      if (contains(last_layer, e.dst)) {
        g.remove_edge(e);
        g.add_edge(DirectedEdge{e.src, sync_node});
        g.add_edge(DirectedEdge{sync_node, e.dst});
      }
    }
    
    for (DirectedEdge const &e : get_edges(g)) {
      if (contains(forest, e.src) && (depth_map.at(e.dst) > depth_map.at(n))) {
        g.remove_edge(e);
        g.add_edge(DirectedEdge{sync_node, e.dst});
      }
    }
    
    g = materialize_digraph_view<AdjacencyDiGraph>(transitive_reduction(g));

  }

  std::cout << get_edges(g) << std::endl;
  std::unordered_set<Node> dummy_nodes =  set_difference(get_nodes(g), original_nodes);
  return get_serial_parallel_decomposition_with_dummy_nodes(g, dummy_nodes).value();
}

SerialParallelDecomposition one_node_at_a_time_spanish_sp_ization(DiGraph g) {
  assert(is_2_terminal_dag(g));
  SerialParallelDecomposition sp =
      one_node_at_a_time_spanish_sp_ization_unchecked(g);
  assert(is_valid_sp_ization(g, sp));
  return sp;
}

} // namespace FlexFlow
