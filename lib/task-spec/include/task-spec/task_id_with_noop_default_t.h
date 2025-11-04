#ifndef _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_TASK_ID_WITH_NOOP_DEFAULT_T_H
#define _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_TASK_ID_WITH_NOOP_DEFAULT_T_H

#include "op-attrs/computation_graph_op_attrs.dtg.h"
#include "op-attrs/operator_type.dtg.h"
#include "task-spec/op_task_id_t.dtg.h"
#include "task-spec/task_id_with_noop_default_t.dtg.h"

namespace FlexFlow {


task_id_with_noop_default_t lift_task_id_t(task_id_t);
task_id_with_noop_default_t default_noop_task();

task_id_with_noop_default_t
  lower_op_task_id_to_task_id_with_noop_default_t(op_task_id_t, 
                                                  ComputationGraphOpAttrs const &);

task_id_with_noop_default_t
  get_init_task_id_for_op_attrs(ComputationGraphOpAttrs const &);

task_id_with_noop_default_t
  get_fwd_task_id_for_op_attrs(ComputationGraphOpAttrs const &);

task_id_with_noop_default_t
  get_bwd_task_id_for_op_attrs(ComputationGraphOpAttrs const &);

} // namespace FlexFlow

#endif
