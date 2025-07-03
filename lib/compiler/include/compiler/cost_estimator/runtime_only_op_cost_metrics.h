#ifndef _FLEXFLOW_LIB_COMPILER_INCLUDE_COMPILER_COST_ESTIMATOR_RUNTIME_ONLY_OP_COST_METRICS_H
#define _FLEXFLOW_LIB_COMPILER_INCLUDE_COMPILER_COST_ESTIMATOR_RUNTIME_ONLY_OP_COST_METRICS_H

#include "compiler/cost_estimator/op_cost_metrics.dtg.h"
#include "compiler/cost_estimator/runtime_only_op_cost_metrics.dtg.h"

namespace FlexFlow {

RuntimeOnlyOpCostMetrics
    runtime_only_from_op_cost_metrics(OpCostMetrics const &);

} // namespace FlexFlow

#endif
