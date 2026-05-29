#include "utils/one_to_many/one_to_many.h"
#include "utils/archetypes/jsonable_value_type.h"
#include "utils/archetypes/rapidcheckable_value_type.h"
#include "utils/archetypes/ordered_value_type.h"
#include "utils/archetypes/jsonable_ordered_value_type.h"

using namespace ::FlexFlow;

namespace FlexFlow {

using L = ordered_value_type<0>;
using R = ordered_value_type<1>;

template struct OneToMany<L, R>;

template std::map<L, nonempty_set<R>>
    format_as(OneToMany<L, R> const &);

template std::ostream &operator<<(std::ostream &, OneToMany<L, R> const &);

template std::unordered_set<std::pair<L, R>>
    unstructured_relation_from_one_to_many(OneToMany<L, R> const &);

} // namespace FlexFlow

namespace nlohmann {

using L = ::FlexFlow::jsonable_ordered_value_type<0>;
using R = ::FlexFlow::jsonable_ordered_value_type<1>;

template struct adl_serializer<::FlexFlow::OneToMany<L, R>>;

} // namespace nlohmann

namespace rc {

using L = ::FlexFlow::rapidcheckable_value_type<0>;
using R = ::FlexFlow::rapidcheckable_value_type<1>;

template struct Arbitrary<::FlexFlow::OneToMany<L, R>>;
template struct Arbitrary<::FlexFlow::OneToMany<int, std::string>>;

} // namespace rc

namespace std {

using L = ::FlexFlow::ordered_value_type<0>;
using R = ::FlexFlow::ordered_value_type<1>;

template struct hash<OneToMany<L, R>>;

} // namespace std
