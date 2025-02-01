#include "utils/containers/enumerate.h"
#include "utils/archetypes/value_type.h"

namespace FlexFlow {

using T = value_type<0>;

template std::map<nonnegative_int, T> enumerate(std::vector<T> const &);

template std::map<nonnegative_int, T> enumerate(std::unordered_set<T> const &);

} // namespace FlexFlow
