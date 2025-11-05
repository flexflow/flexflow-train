#ifndef _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_OPS_IMPL_INPUT_H
#define _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_OPS_IMPL_INPUT_H

#include "op-attrs/ops/input_attrs.dtg.h"
#include "task-spec/op_task_invocation.h"

namespace FlexFlow {

std::unordered_set<task_id_t> get_task_ids(InputAttrs const &);

} // namespace FlexFlow

#endif
