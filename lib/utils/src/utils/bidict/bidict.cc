#include "utils/bidict/bidict.h"
#include "utils/archetypes/jsonable_ordered_value_type.h"
#include "utils/archetypes/ordered_value_type.h"
#include "utils/archetypes/rapidcheckable_value_type.h"

namespace FlexFlow {

using L = jsonable_ordered_value_type<0>;
using R = jsonable_ordered_value_type<1>;

template struct bidict<L, R>;

template std::map<L, R> format_as(bidict<L, R> const &);

template std::ostream &operator<<(std::ostream &, bidict<L, R> const &);

} // namespace FlexFlow

namespace nlohmann {

using L = ::FlexFlow::jsonable_ordered_value_type<0>;
using R = ::FlexFlow::jsonable_ordered_value_type<1>;

template struct adl_serializer<::FlexFlow::bidict<L, R>>;

} // namespace nlohmann

namespace rc {

using L = ::FlexFlow::rapidcheckable_value_type<0>;
using R = ::FlexFlow::rapidcheckable_value_type<1>;

template struct Arbitrary<::FlexFlow::bidict<L, R>>;
template struct Arbitrary<::FlexFlow::bidict<int, std::string>>;

} // namespace rc

namespace std {

using L = ::FlexFlow::ordered_value_type<0>;
using R = ::FlexFlow::ordered_value_type<1>;

template struct hash<::FlexFlow::bidict<L, R>>;

} // namespace std
