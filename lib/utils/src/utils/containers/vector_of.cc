#include "utils/containers/vector_of.h"
#include "utils/archetypes/value_type.h"
#include <set>
#include <unordered_set>

namespace FlexFlow {

using T = value_type<0>;

template std::vector<T> vector_of(std::vector<T> const &);

template std::vector<T> vector_of(std::unordered_set<T> const &);

template std::vector<T> vector_of(std::set<T> const &);

template std::vector<T> vector_of(std::optional<T> const &);

} // namespace FlexFlow
