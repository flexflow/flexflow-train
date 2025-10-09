#ifndef _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_FWB_OP_TASK_TYPE_H
#define _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_FWB_OP_TASK_TYPE_H

#include "task-spec/fwb_op_task_type.dtg.h"
#include "task-spec/op_task_type.dtg.h"
#include <optional>

namespace FlexFlow {

std::optional<OpTaskType>
  op_task_type_from_fwb_op_task_type(FwbOpTaskType);

} // namespace FlexFlow

#endif
