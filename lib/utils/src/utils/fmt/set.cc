#include "utils/fmt/set.h"
#include "utils/archetypes/ordered_value_type.h"

using T = ::FlexFlow::ordered_value_type<0>;

namespace fmt {

template struct formatter<::std::set<T>, char>;

}

namespace FlexFlow {

template std::ostream &operator<<(std::ostream &, std::set<T> const &);

}
