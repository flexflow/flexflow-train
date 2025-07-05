#ifndef _FLEXFLOW_WEIGHT_H
#define _FLEXFLOW_WEIGHT_H

#include "op-attrs/ops/weight_attrs.dtg.h"
#include "task-spec/op_task_invocation.h"

namespace FlexFlow {

std::vector<task_id_t> get_task_ids(WeightAttrs const &);

} // namespace FlexFlow

#endif
