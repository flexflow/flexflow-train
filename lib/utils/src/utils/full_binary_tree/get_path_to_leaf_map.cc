#include "utils/full_binary_tree/get_path_to_leaf_map.h"
#include "utils/archetypes/value_type.h"

namespace FlexFlow {

using Tree = value_type<0>;
using Parent = value_type<1>;
using Leaf = value_type<2>;

template
  std::unordered_map<BinaryTreePath, Leaf>
      get_path_to_leaf_map(Tree const &,
                 FullBinaryTreeImplementation<Tree, Parent, Leaf> const &);

} // namespace FlexFlow
