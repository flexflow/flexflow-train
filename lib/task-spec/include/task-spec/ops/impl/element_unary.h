#ifndef _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_OPS_IMPL_ELEMENT_UNARY_H
#define _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_OPS_IMPL_ELEMENT_UNARY_H

#include "op-attrs/ops/element_unary_attrs.dtg.h"
#include "task-spec/task_impl_function.dtg.h"

namespace FlexFlow {

TaskImplFunction get_element_unary_init_task_impl();
TaskImplFunction get_element_unary_fwd_task_impl();
TaskImplFunction get_element_unary_bwd_task_impl();

} // namespace FlexFlow

#endif
