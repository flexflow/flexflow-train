#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_FULL_BINARY_TREE_GET_PATH_TO_LEAF_MAP_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_FULL_BINARY_TREE_GET_PATH_TO_LEAF_MAP_H

#include "utils/containers/merge_maps.h"
#include "utils/containers/multiset_union.h"
#include "utils/containers/map_keys.h"
#include "utils/full_binary_tree/binary_tree_path.dtg.h"
#include "utils/full_binary_tree/binary_tree_path.h"
#include "utils/full_binary_tree/full_binary_tree_visitor.dtg.h"
#include "utils/full_binary_tree/visit.h"
#include <unordered_set>

namespace FlexFlow {

template <typename Tree, typename Parent, typename Leaf>
std::unordered_map<BinaryTreePath, Leaf>
    get_path_to_leaf_map(Tree const &tree,
               FullBinaryTreeImplementation<Tree, Parent, Leaf> const &impl) {

  auto visitor =
      FullBinaryTreeVisitor<std::unordered_map<BinaryTreePath, Leaf>, Tree, Parent, Leaf>{
          [&](Parent const &parent) -> std::unordered_map<BinaryTreePath, Leaf> {

            std::unordered_map<BinaryTreePath, Leaf> left_map = 
                map_keys(get_path_to_leaf_map(impl.get_left_child(parent), impl),
                         [](BinaryTreePath const &p) {
                           return nest_inside_left_child(p);
                         });

            std::unordered_map<BinaryTreePath, Leaf> right_map = 
                map_keys(get_path_to_leaf_map(impl.get_right_child(parent), impl),
                         [](BinaryTreePath const &p) {
                           return nest_inside_right_child(p);
                         });

            return merge_disjoint_maps(left_map, right_map);
          },
          [](Leaf const &leaf) -> std::unordered_map<BinaryTreePath, Leaf> {
            return std::unordered_map<BinaryTreePath, Leaf>{
              {binary_tree_root_path(), leaf},
            };
          },
      };

  return visit(tree, impl, visitor);
}

} // namespace FlexFlow

#endif
