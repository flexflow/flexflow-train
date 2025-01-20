#include "utils/stack_vector/stack_vector_of.h"
#include "utils/archetypes/ordered_value_type.h"

namespace FlexFlow {

using T = ordered_value_type<0>;

template stack_vector<T, 5> stack_vector_of<5>(std::vector<T> const &vector);

} // namespace FlexFlow
