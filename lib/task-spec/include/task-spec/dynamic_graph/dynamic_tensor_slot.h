#ifndef _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_DYNAMIC_GRAPH_DYNAMIC_TENSOR_SLOT_H
#define _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_DYNAMIC_GRAPH_DYNAMIC_TENSOR_SLOT_H

#include "task-spec/dynamic_graph/dynamic_tensor_slot.dtg.h"

namespace FlexFlow {

DynamicTensorSlot decide_tensor_slot_role(DynamicTensorSlot const &,
                                          DynamicTensorRole);

} // namespace FlexFlow

#endif
