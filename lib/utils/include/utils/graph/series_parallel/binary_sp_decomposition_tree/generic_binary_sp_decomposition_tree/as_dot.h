#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_SERIES_PARALLEL_BINARY_SP_DECOMPOSITION_TREE_GENERIC_BINARY_SP_DECOMPOSITION_TREE_AS_DOT_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_SERIES_PARALLEL_BINARY_SP_DECOMPOSITION_TREE_GENERIC_BINARY_SP_DECOMPOSITION_TREE_AS_DOT_H

#include "utils/full_binary_tree/as_dot.h"
#include "utils/graph/series_parallel/binary_sp_decomposition_tree/generic_binary_sp_decomposition_tree/generic_binary_sp_decomposition_tree_implementation.dtg.h"
#include "utils/graph/series_parallel/binary_sp_decomposition_tree/generic_binary_sp_decomposition_tree/generic_binary_sp_decomposition_tree_implementation.h"
#include "utils/overload.h"

namespace FlexFlow {

template <typename Tree, typename Series, typename Parallel, typename Leaf>
std::string as_dot(Tree const &tree, 
                   GenericBinarySPDecompositionTreeImplementation<Tree, Series, Parallel, Leaf> const &impl,
                   std::function<std::string(Series const &)> const &get_series_label,
                   std::function<std::string(Parallel const &)> const &get_parallel_label,
                   std::function<std::string(Leaf const &)> const &get_leaf_label) {
  FullBinaryTreeImplementation<Tree, std::variant<Series, Parallel>, Leaf> full_binary_tree_impl 
    = get_full_binary_impl_from_generic_sp_impl(impl);

  std::function<std::string(std::variant<Series, Parallel> const &)> get_parent_label 
    = [&](std::variant<Series, Parallel> const &parent) -> std::string {
      return std::visit(overload {
                          [&](Series const &series) -> std::string {
                            return get_series_label(series);
                          },
                          [&](Parallel const &parallel) -> std::string {
                            return get_parallel_label(parallel);
                          },
                        }, parent);
    };

  return as_dot(tree, full_binary_tree_impl, get_parent_label, get_leaf_label);
}

} // namespace FlexFlow

#endif
