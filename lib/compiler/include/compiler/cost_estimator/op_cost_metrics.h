#ifndef _FLEXFLOW_LIB_COMPILER_INCLUDE_COMPILER_COST_ESTIMATOR_OP_COST_METRICS_H
#define _FLEXFLOW_LIB_COMPILER_INCLUDE_COMPILER_COST_ESTIMATOR_OP_COST_METRICS_H

#include "compiler/cost_estimator/op_cost_metrics.dtg.h"
#include "compiler/cost_estimator/runtime_only_op_cost_metrics.dtg.h"

namespace FlexFlow {

OpCostMetrics make_op_cost_metrics_from_runtime_only(
    RuntimeOnlyOpCostMetrics const &runtime_only,
    num_bytes_t const &memory_usage);

} // namespace FlexFlow

#endif
