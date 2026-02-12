#ifndef _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_DYNAMIC_GRAPH_LOSS_INSERTION_H
#define _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_DYNAMIC_GRAPH_LOSS_INSERTION_H

#include "op-attrs/ops/loss_functions/loss_attrs.dtg.h"
#include "task-spec/dynamic_graph/dynamic_open_dataflow_graph.dtg.h"
#include "task-spec/dynamic_graph/dynamic_tensor_guid_t.dtg.h"
#include "task-spec/dynamic_graph/dynamic_value_attrs.dtg.h"
#include "task-spec/dynamic_graph/loss_insertion_result.dtg.h"

namespace FlexFlow {

LossInsertionResult perform_loss_insertion(DynamicOpenDataflowGraph const &dg,
                                           LossAttrs const &loss_attrs,
                                           dynamic_tensor_guid_t logit_tensor);

} // namespace FlexFlow

#endif
