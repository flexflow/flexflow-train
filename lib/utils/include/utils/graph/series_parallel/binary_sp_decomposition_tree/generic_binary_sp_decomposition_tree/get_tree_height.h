#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_SERIES_PARALLEL_BINARY_SP_DECOMPOSITION_TREE_GENERIC_BINARY_SP_DECOMPOSITION_TREE_GET_TREE_HEIGHT_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_SERIES_PARALLEL_BINARY_SP_DECOMPOSITION_TREE_GENERIC_BINARY_SP_DECOMPOSITION_TREE_GET_TREE_HEIGHT_H

#include "utils/full_binary_tree/get_tree_height.h"
#include "utils/graph/series_parallel/binary_sp_decomposition_tree/generic_binary_sp_decomposition_tree/generic_binary_sp_decomposition_tree_implementation.h"

namespace FlexFlow {

template <typename Tree, typename Series, typename Parallel, typename Leaf>
nonnegative_int get_tree_height(
    Tree const &tree,
    GenericBinarySPDecompositionTreeImplementation<Tree,
                                                   Series,
                                                   Parallel,
                                                   Leaf> const &impl) {

  FullBinaryTreeImplementation<Tree, std::variant<Series, Parallel>, Leaf>
      full_binary_impl = get_full_binary_impl_from_generic_sp_impl(impl);

  return get_tree_height(tree, full_binary_impl);
}


} // namespace FlexFlow

#endif
