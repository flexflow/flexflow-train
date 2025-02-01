#include "utils/containers/repeat_element.h"
#include "utils/archetypes/value_type.h"

namespace FlexFlow {

using T = value_type<0>;

template std::vector<T> repeat_element(nonnegative_int, T const &);

} // namespace FlexFlow
