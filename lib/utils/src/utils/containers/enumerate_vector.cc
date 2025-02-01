#include "utils/containers/enumerate_vector.h"
#include "utils/archetypes/value_type.h"

namespace FlexFlow {

using T = value_type<0>;

template std::map<nonnegative_int, T> enumerate_vector(std::vector<T> const &);

} // namespace FlexFlow
