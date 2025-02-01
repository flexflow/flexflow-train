#include "utils/nonnegative_int/num_elements.h"
#include "utils/archetypes/value_type.h"

namespace FlexFlow {

using E = value_type<0>;

template nonnegative_int num_elements(std::vector<E> const &);

} // namespace FlexFlow
