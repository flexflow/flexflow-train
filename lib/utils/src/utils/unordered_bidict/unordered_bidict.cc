#include "utils/unordered_bidict/unordered_bidict.h"
#include "utils/archetypes/jsonable_ordered_value_type.h"
#include "utils/archetypes/value_type.h"
#include "utils/archetypes/rapidcheckable_value_type.h"

namespace FlexFlow {

using L = jsonable_ordered_value_type<0>;
using R = jsonable_ordered_value_type<1>;

template struct unordered_bidict<L, R>;

template std::unordered_map<L, R> format_as(unordered_bidict<L, R> const &);

template std::ostream &operator<<(std::ostream &, unordered_bidict<L, R> const &);

} // namespace FlexFlow

namespace nlohmann {

using L = ::FlexFlow::jsonable_ordered_value_type<0>;
using R = ::FlexFlow::jsonable_ordered_value_type<1>;

template struct adl_serializer<::FlexFlow::unordered_bidict<L, R>>;

} // namespace nlohmann

namespace rc {

using L = ::FlexFlow::rapidcheckable_value_type<0>;
using R = ::FlexFlow::rapidcheckable_value_type<1>;

template struct Arbitrary<::FlexFlow::unordered_bidict<L, R>>;
template struct Arbitrary<::FlexFlow::unordered_bidict<int, std::string>>;

} // namespace rc

namespace std {

using L = ::FlexFlow::value_type<0>;
using R = ::FlexFlow::value_type<1>;

template struct hash<::FlexFlow::unordered_bidict<L, R>>;

} // namespace std
