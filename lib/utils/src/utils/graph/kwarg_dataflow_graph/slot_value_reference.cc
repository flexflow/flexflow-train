#include "utils/graph/kwarg_dataflow_graph/slot_value_reference.h"
#include "utils/archetypes/value_type.h"

namespace FlexFlow {

using SlotName = value_type<0>;

template
  SlotName get_slot_name_for_slot_value_reference(SlotValueReference<SlotName> const &);


} // namespace FlexFlow
