#include "utils/containers/subvec.h"
#include "utils/archetypes/value_type.h"

namespace FlexFlow {

using T = value_type<0>;

template 
  std::vector<T> subvec(std::vector<T> const &,
                        std::optional<int> const &,
                        std::optional<int> const &);

} // namespace FlexFlow
