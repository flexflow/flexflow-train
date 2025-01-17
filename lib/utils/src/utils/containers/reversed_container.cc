#include "utils/containers/reversed_container.h"
#include "utils/archetypes/value_type.h"
#include <vector>

namespace FlexFlow {

using T = value_type<0>;
using C = std::vector<T>;

template 
  reversed_container_t<C> reversed_container(C const &);

} // namespace FlexFlow
