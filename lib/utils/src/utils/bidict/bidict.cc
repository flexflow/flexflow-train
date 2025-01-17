#include "utils/bidict/bidict.h"
#include "utils/archetypes/value_type.h"

namespace FlexFlow {

using L = value_type<0>;
using R = value_type<1>;

template struct bidict<L, R>;

template
  std::unordered_map<L, R> format_as(bidict<L, R> const &);

template
  std::ostream &operator<<(std::ostream &, bidict<L, R> const &);

} // namespace FlexFlow
