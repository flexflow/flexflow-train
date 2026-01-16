#ifndef _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_OPTIMIZER_H
#define _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_OPTIMIZER_H

#include "pcg/optimizer_attrs.dtg.h"
#include "pcg/optimizers/adam_optimizer_attrs.dtg.h"
#include "pcg/optimizers/sgd_optimizer_attrs.dtg.h"
#include "task-spec/task_impl_function.dtg.h"

namespace FlexFlow {

TaskImplFunction get_update_task_impl(OptimizerAttrs const &);
TaskImplFunction get_sgd_update_task_impl();
TaskImplFunction get_adam_update_task_impl();

} // namespace FlexFlow

#endif
