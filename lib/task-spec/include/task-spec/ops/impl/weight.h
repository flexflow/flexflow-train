#ifndef _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_OPS_IMPL_WEIGHT_H
#define _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_OPS_IMPL_WEIGHT_H

#include "op-attrs/ops/weight_attrs.dtg.h"
#include "task-spec/op_task_invocation.h"

namespace FlexFlow {

std::unordered_set<task_id_t> get_task_ids(WeightAttrs const &);

} // namespace FlexFlow

#endif
