#include "utils/one_to_many/one_to_many.h"
#include "utils/archetypes/value_type.h"

using namespace ::FlexFlow;

using L = value_type<0>;
using R = value_type<1>;

namespace FlexFlow {

template struct OneToMany<L, R>;

template std::unordered_map<L, std::unordered_set<R>> format_as(OneToMany<L, R> const &);

template std::ostream &operator<<(std::ostream &, OneToMany<L, R> const &);

} // namespace FlexFlow

namespace std {

template struct hash<OneToMany<L, R>>;

}

