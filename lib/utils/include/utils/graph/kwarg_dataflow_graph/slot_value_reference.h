#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_KWARG_DATAFLOW_GRAPH_SLOT_VALUE_REFERENCE_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_KWARG_DATAFLOW_GRAPH_SLOT_VALUE_REFERENCE_H

#include "utils/graph/kwarg_dataflow_graph/slot_value_reference.dtg.h"
#include "utils/overload.h"

namespace FlexFlow {

template <typename SlotName>
SlotName get_slot_name_for_slot_value_reference(SlotValueReference<SlotName> const &ref) {
  return ref.template visit<SlotName>(overload {
    [](SlotName const &n) {
      return n; 
    },
    [](VariadicSlotValueReference<SlotName> const &v) {
      return v.slot_name; 
    }
  });
}

} // namespace FlexFlow

#endif
