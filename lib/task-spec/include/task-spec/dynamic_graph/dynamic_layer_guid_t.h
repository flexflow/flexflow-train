#ifndef _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_DYNAMIC_GRAPH_DYNAMIC_LAYER_GUID_T_H
#define _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_DYNAMIC_GRAPH_DYNAMIC_LAYER_GUID_T_H

#include "pcg/layer_guid_t.dtg.h"
#include "pcg/parallel_computation_graph/parallel_layer_guid_t.dtg.h"
#include "task-spec/dynamic_graph/dynamic_layer_guid_t.dtg.h"

namespace FlexFlow {

dynamic_layer_guid_t mk_dynamic_layer_guid(layer_guid_t);
dynamic_layer_guid_t mk_dynamic_layer_guid_parallel(parallel_layer_guid_t);
dynamic_layer_guid_t mk_dynamic_layer_guid_loss();

} // namespace FlexFlow

#endif
