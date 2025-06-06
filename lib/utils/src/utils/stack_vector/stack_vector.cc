#include "utils/stack_vector/stack_vector.h"
#include "utils/archetypes/value_type.h"

namespace FlexFlow {

using T = value_type<0>;

template struct stack_vector<T, 5>;
template struct stack_vector<int, 5>;

} // namespace FlexFlow
