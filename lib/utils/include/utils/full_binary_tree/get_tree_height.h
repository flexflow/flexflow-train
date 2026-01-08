#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_FULL_BINARY_TREE_GET_TREE_HEIGHT_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_FULL_BINARY_TREE_GET_TREE_HEIGHT_H

#include "utils/full_binary_tree/full_binary_tree_implementation.dtg.h"
#include "utils/full_binary_tree/visit.h"
#include "utils/nonnegative_int/nonnegative_int.h"

namespace FlexFlow {

template <typename Tree, typename Parent, typename Leaf>
nonnegative_int get_tree_height(
    Tree const &tree,
    FullBinaryTreeImplementation<Tree, Parent, Leaf> const &impl) {

  auto visitor = FullBinaryTreeVisitor<nonnegative_int, Tree, Parent, Leaf>{
      [&](Parent const &parent) -> nonnegative_int {
        nonnegative_int left_height =
            get_tree_height(impl.get_left_child(parent), impl);
        nonnegative_int right_height =
            get_tree_height(impl.get_right_child(parent), impl);
        return std::max(left_height, right_height) + 1_n;
      },
      [](Leaf const &) -> nonnegative_int { return 0_n; },
  };

  return visit(tree, impl, visitor);
}

} // namespace FlexFlow

#endif
