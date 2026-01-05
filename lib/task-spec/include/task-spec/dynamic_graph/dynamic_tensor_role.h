#ifndef _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_DYNAMIC_GRAPH_DYNAMIC_TENSOR_ROLE_H
#define _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_DYNAMIC_GRAPH_DYNAMIC_TENSOR_ROLE_H

#include "task-spec/dynamic_graph/dynamic_tensor_role.dtg.h"
#include "pcg/optimizer_slot_name.dtg.h"

namespace FlexFlow {

DynamicTensorRole dynamic_tensor_role_from_fwb_tensor_type(FwbTensorType);

DynamicTensorRole mk_dynamic_tensor_role_fwd();
DynamicTensorRole mk_dynamic_tensor_role_bwd();
DynamicTensorRole mk_dynamic_tensor_role_opt(OptimizerSlotName);

} // namespace FlexFlow

#endif
