#ifndef _FLEXFLOW_INPUT_H
#define _FLEXFLOW_INPUT_H

#include "op-attrs/ops/input_attrs.dtg.h"
#include "task-spec/op_task_invocation.h"

namespace FlexFlow {

std::vector<task_id_t> get_task_ids(InputAttrs const &);

} // namespace FlexFlow

#endif
