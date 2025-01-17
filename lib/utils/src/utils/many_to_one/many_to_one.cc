#include "utils/many_to_one/many_to_one.h"
#include "utils/archetypes/value_type.h"

using namespace ::FlexFlow;

using L = value_type<0>;
using R = value_type<1>;

namespace FlexFlow {

template struct ManyToOne<L, R>;

template std::unordered_map<std::unordered_set<L>, R> format_as(ManyToOne<L, R> const &);

template std::ostream &operator<<(std::ostream &, ManyToOne<L, R> const &);

} // namespace FlexFlow

namespace std {

template struct hash<ManyToOne<L, R>>;

}
