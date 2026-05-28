#include "utils/many_to_one/many_to_one.h"
#include "utils/archetypes/jsonable_ordered_value_type.h"
#include "utils/archetypes/rapidcheckable_value_type.h"
#include "utils/archetypes/value_type.h"

using namespace ::FlexFlow;

namespace FlexFlow {

using L = value_type<0>;
using R = value_type<1>;

template struct ManyToOne<L, R>;

template std::unordered_map<std::unordered_set<L>, R>
    format_as(ManyToOne<L, R> const &);

template std::ostream &operator<<(std::ostream &, ManyToOne<L, R> const &);

template std::unordered_set<std::pair<L, R>>
    unstructured_relation_from_many_to_one(ManyToOne<L, R> const &);

template ManyToOne<L, R> many_to_one_from_unstructured_relation(
    std::unordered_set<std::pair<L, R>> const &);

} // namespace FlexFlow

namespace nlohmann {

using L = ::FlexFlow::jsonable_ordered_value_type<0>;
using R = ::FlexFlow::jsonable_ordered_value_type<1>;

template struct adl_serializer<::FlexFlow::ManyToOne<L, R>>;

} // namespace nlohmann

namespace rc {

using L = ::FlexFlow::rapidcheckable_value_type<0>;
using R = ::FlexFlow::rapidcheckable_value_type<1>;

template struct Arbitrary<::FlexFlow::ManyToOne<L, R>>;
template struct Arbitrary<::FlexFlow::ManyToOne<int, std::string>>;

} // namespace rc

namespace std {

using L = ::FlexFlow::value_type<0>;
using R = ::FlexFlow::value_type<1>;

template struct hash<ManyToOne<L, R>>;

} // namespace std
