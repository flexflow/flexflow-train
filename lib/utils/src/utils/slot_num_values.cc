#include "utils/slot_num_values.h"
#include "utils/archetypes/value_type.h"

namespace FlexFlow {

SlotNumValues slot_num_values_singular() {
  return SlotNumValues{std::monostate{}};
}

SlotNumValues slot_num_values_variadic(positive_int n) {
  return SlotNumValues{n};
}

template SlotNumValues get_slot_num_values(SingularOrVariadic<value_type<0>> const &);

} // namespace FlexFlow
