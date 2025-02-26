#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_FULL_BINARY_TREE_AS_DOT_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_FULL_BINARY_TREE_AS_DOT_H

#include "utils/containers/get_only.h"
#include "utils/dot_file.h"
#include "utils/full_binary_tree/full_binary_tree_implementation.dtg.h"
#include "utils/full_binary_tree/full_binary_tree_visitor.dtg.h"
#include "utils/graph/dataflow_graph/dataflow_graph.h"
#include "utils/graph/dataflow_graph/dataflow_graph_view.h"
#include "utils/graph/digraph/digraph_view.h"
#include "utils/graph/instances/adjacency_digraph.h"
#include "utils/graph/instances/unordered_set_dataflow_graph.h"
#include "utils/graph/instances/unordered_set_labelled_open_dataflow_graph.h"
#include "utils/graph/labelled_dataflow_graph/algorithms/view_as_labelled_open_dataflow_graph.h"
#include "utils/graph/labelled_dataflow_graph/labelled_dataflow_graph.h"
#include <string>
#include <functional>
#include <sstream>
#include "utils/graph/labelled_open_dataflow_graph/algorithms/as_dot.h"
#include "utils/full_binary_tree/visit.h"

namespace FlexFlow {

template <typename Tree, typename Parent, typename Leaf, typename NodeLabel>
LabelledDataflowGraph<NodeLabel, std::monostate> as_labelled_dataflow_graph(Tree const &tree,
                             FullBinaryTreeImplementation<Tree, Parent, Leaf> const &impl,
                             std::function<NodeLabel(Parent const &)> const &get_parent_label,
                             std::function<NodeLabel(Leaf const &)> const &get_leaf_label) {
  auto g = LabelledDataflowGraph<NodeLabel, std::monostate>
    ::template create<UnorderedSetLabelledOpenDataflowGraph<NodeLabel, std::monostate>>();

  FullBinaryTreeVisitor<DataflowOutput, Tree, Parent, Leaf> visitor = FullBinaryTreeVisitor<DataflowOutput, Tree, Parent, Leaf>{
    [&](Parent const &parent) -> DataflowOutput {
      DataflowOutput left_child_output = visit(impl.get_left_child(parent), impl, visitor);
      DataflowOutput right_child_output = visit(impl.get_right_child(parent), impl, visitor);
      NodeLabel parent_label = get_parent_label(parent);
      NodeAddedResult parent_added = g.add_node(parent_label, {left_child_output, right_child_output}, {std::monostate{}});
      return get_only(parent_added.outputs);
    },
    [&](Leaf const &leaf) -> DataflowOutput {
      NodeLabel leaf_label = get_leaf_label(leaf);
      NodeAddedResult leaf_added = g.add_node(leaf_label, {}, {std::monostate{}});
      return get_only(leaf_added.outputs);
    },
  };

  visit(tree, impl, visitor);

  return g;
}

template <typename Tree, typename Parent, typename Leaf>
std::string as_dot(Tree const &tree,
                   FullBinaryTreeImplementation<Tree, Parent, Leaf> const &impl,
                   std::function<std::string(Parent const &)> const &get_parent_label,
                   std::function<std::string(Leaf const &)> const &get_leaf_label) {

  LabelledDataflowGraphView<std::string, std::monostate> g = as_labelled_dataflow_graph(tree, impl, get_parent_label, get_leaf_label);

  std::function<std::string(std::string const &)> get_node_label = [](std::string const &s) { return s; };
  std::function<std::string(std::monostate const &)> get_input_label = [](std::monostate const &) { return ""; };

  return as_dot(view_as_labelled_open_dataflow_graph(g), 
                get_node_label,
                get_input_label);
}

} // namespace FlexFlow

#endif
