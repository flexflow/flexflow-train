#include "utils/full_binary_tree/as_dot.h"
#include "utils/archetypes/value_type.h"

namespace FlexFlow {

using Tree = value_type<0>;
using Parent = value_type<1>;
using Leaf = value_type<2>;

template
  std::string as_dot(Tree const &,
                     FullBinaryTreeImplementation<Tree, Parent, Leaf> const &,
                     std::function<std::string(Parent const &)> const &,
                     std::function<std::string(Leaf const &)> const &);

} // namespace FlexFlow
