#ifndef _FLEXFLOW_LIB_COMPILER_INCLUDE_COMPILER_COST_ESTIMATOR_RUNTIME_ONLY_OP_COST_ESTIMATE_KEY_H
#define _FLEXFLOW_LIB_COMPILER_INCLUDE_COMPILER_COST_ESTIMATOR_RUNTIME_ONLY_OP_COST_ESTIMATE_KEY_H

#include "compiler/cost_estimator/runtime_only_op_cost_estimate_key.dtg.h"
#include "pcg/parallel_computation_graph/parallel_computation_graph.dtg.h"
#include "pcg/parallel_computation_graph/parallel_layer_guid_t.dtg.h"

namespace FlexFlow {

RuntimeOnlyOpCostEstimateKey get_mapped_runtime_only_op_cost_estimate_key_for_layer(
    ParallelComputationGraph const &pcg,
    parallel_layer_guid_t const &parallel_layer_guid,
    MachineView const &machine_view);

} // namespace FlexFlow

#endif
