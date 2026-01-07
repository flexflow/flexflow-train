#ifndef _FLEXFLOW_UTILS_GRAPH_ALGORITHMS_H
#define _FLEXFLOW_UTILS_GRAPH_ALGORITHMS_H

#include "utils/dot_file.h"
#include "utils/graph/digraph/digraph.h"
#include "utils/graph/graph_split.dtg.h"
#include "utils/graph/node/graph.h"
#include "utils/graph/undirected/undirected_graph.h"

namespace FlexFlow {

std::vector<Node> add_nodes(Graph &, int);
std::vector<Node> add_nodes(UndirectedGraph &, int);
std::vector<Node> add_nodes(DiGraph &, int);

std::unordered_set<Node> query_nodes(GraphView const &,
                                     std::unordered_set<Node> const &);

void remove_node(DiGraph &, Node const &);
void remove_node(UndirectedGraph &, Node const &);

void remove_node_if_unused(DiGraph &, Node const &);
void remove_node_if_unused(UndirectedGraph &, Node const &);

bool empty(GraphView const &);

void add_edges(DiGraph &, std::vector<DirectedEdge> const &);
void add_edges(UndirectedGraph &, std::vector<UndirectedEdge> const &);

bool contains_node(GraphView const &, Node const &);

bool contains_edge(DiGraphView const &, DirectedEdge const &);
bool contains_edge(UndirectedGraphView const &, UndirectedEdge const &);

void remove_edges(DiGraph &, std::unordered_set<DirectedEdge> const &);
void remove_edges(UndirectedGraph &, std::vector<UndirectedEdge> const &);

std::unordered_set<UndirectedEdge> get_edges(UndirectedGraphView const &);

std::unordered_set<UndirectedEdge> get_node_edges(UndirectedGraphView const &,
                                                  Node const &);

std::unordered_set<UndirectedEdge> get_node_edges(UndirectedGraphView const &,
                                                  Node const &);
std::unordered_set<UndirectedEdge>
    get_node_edges(UndirectedGraphView const &,
                   std::unordered_set<Node> const &);

std::unordered_set<Node> get_neighbors(UndirectedGraphView const &,
                                       Node const &);
std::unordered_set<Node> get_neighbors(DiGraphView const &, Node const &);

std::vector<Node>
    get_dfs_ordering(DiGraphView const &,
                     std::unordered_set<Node> const &starting_points);
std::vector<Node>
    get_unchecked_dfs_ordering(DiGraphView const &,
                               std::unordered_set<Node> const &starting_points);
std::vector<Node>
    get_bfs_ordering(DiGraphView const &,
                     std::unordered_set<Node> const &starting_points);
std::vector<Node> get_unchecked_topological_ordering(DiGraphView const &);

std::unordered_set<DirectedEdge>
    get_transitive_reduction_delta(DiGraphView const &);

UndirectedGraphView get_subgraph(UndirectedGraphView const &,
                                 std::unordered_set<Node> const &);
DiGraphView get_subgraph(DiGraphView const &, std::unordered_set<Node> const &);

DiGraphView join(DiGraphView const &lhs, DiGraphView const &rhs);
UndirectedGraphView join(UndirectedGraphView const &lhs,
                         UndirectedGraphView const &rhs);

DiGraphView flipped(DiGraphView const &);

DiGraphView with_added_edges(DiGraphView const &,
                             std::unordered_set<DirectedEdge> const &);

UndirectedGraphView as_undirected(DiGraphView const &);
DiGraphView as_digraph(UndirectedGraphView const &);

void export_as_dot(
    DotFile<Node> &,
    DiGraphView const &,
    std::function<RecordFormatter(Node const &)> const &,
    std::optional<std::function<std::string(DirectedEdge const &)>> =
        std::nullopt);

} // namespace FlexFlow

#endif
