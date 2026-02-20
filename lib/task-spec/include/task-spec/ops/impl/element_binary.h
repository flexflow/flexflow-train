#ifndef _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_OPS_IMPL_ELEMENT_BINARY_H
#define _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_OPS_IMPL_ELEMENT_BINARY_H

#include "op-attrs/ops/element_binary_attrs.dtg.h"
#include "task-spec/task_impl_function.dtg.h"

namespace FlexFlow {

TaskImplFunction get_element_binary_init_task_impl();
TaskImplFunction get_element_binary_fwd_task_impl();
TaskImplFunction get_element_binary_bwd_task_impl();

} // namespace FlexFlow

#endif
