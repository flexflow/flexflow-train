#ifndef _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_DYNAMIC_GRAPH_PARALLEL_OP_UTILS_H
#define _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_DYNAMIC_GRAPH_PARALLEL_OP_UTILS_H

#include "op-attrs/ops/combine.h"
#include "op-attrs/ops/reduction.h"
#include "op-attrs/ops/repartition.h"
#include "op-attrs/ops/replicate.h"
#include "op-attrs/pcg_operator_attrs.dtg.h"
#include "task-spec/dynamic_graph/dynamic_node_attrs.dtg.h"
#include "task-spec/dynamic_graph/training_operation_attrs.dtg.h"

namespace FlexFlow {

inline bool is_parallel_op_attrs(DynamicNodeAttrs const &n) {
  if (!n.op_attrs.has_value()) {
    return false;
  }
  if (!n.op_attrs.value().has<PCGOperatorAttrs>()) {
    return false;
  }
  PCGOperatorAttrs pcg = n.op_attrs.value().get<PCGOperatorAttrs>();
  return pcg.has<ReplicateAttrs>() || pcg.has<RepartitionAttrs>() ||
         pcg.has<CombineAttrs>() || pcg.has<ReductionAttrs>();
}

} // namespace FlexFlow

#endif // _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_DYNAMIC_GRAPH_PARALLEL_OP_UTILS_H
