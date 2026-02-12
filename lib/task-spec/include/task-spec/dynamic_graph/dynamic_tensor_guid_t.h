#ifndef _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_DYNAMIC_GRAPH_DYNAMIC_TENSOR_GUID_T_H
#define _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_DYNAMIC_GRAPH_DYNAMIC_TENSOR_GUID_T_H

#include "pcg/parallel_computation_graph/parallel_tensor_guid_t.dtg.h"
#include "pcg/tensor_guid_t.dtg.h"
#include "task-spec/dynamic_graph/dynamic_tensor_guid_t.dtg.h"

namespace FlexFlow {

dynamic_tensor_guid_t mk_dynamic_tensor_guid_for_tensor_guid(tensor_guid_t);
dynamic_tensor_guid_t
    mk_dynamic_tensor_guid_for_parallel_tensor_guid(parallel_tensor_guid_t);
dynamic_tensor_guid_t mk_dynamic_tensor_guid_for_loss();

} // namespace FlexFlow

#endif
