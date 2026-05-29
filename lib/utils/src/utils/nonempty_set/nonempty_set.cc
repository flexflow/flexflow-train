#include "utils/nonempty_set/nonempty_set.h"
#include "utils/archetypes/ordered_value_type.h"
#include "utils/archetypes/jsonable_ordered_value_type.h"

using T = ::FlexFlow::ordered_value_type<0>;
using J = ::FlexFlow::jsonable_ordered_value_type<0>;

namespace FlexFlow {

template struct nonempty_set<T>;

template bool operator==(std::set<T> const &, nonempty_set<T> const &);

template bool operator!=(std::set<T> const &, nonempty_set<T> const &);

template std::set<T> format_as(nonempty_set<T> const &);
template std::ostream &operator<<(std::ostream &, nonempty_set<T> const &);

} // namespace FlexFlow

namespace nlohmann {

template struct adl_serializer<::FlexFlow::nonempty_set<J>>;

} // namespace nlohmann

namespace std {

template struct hash<::FlexFlow::nonempty_set<T>>;

} // namespace std
