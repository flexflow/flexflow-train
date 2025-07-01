#ifndef _FLEXFLOW_LIB_COMPILER_INCLUDE_COMPILER_COST_ESTIMATOR_OP_COST_ESTIMATE_KEY_H
#define _FLEXFLOW_LIB_COMPILER_INCLUDE_COMPILER_COST_ESTIMATOR_OP_COST_ESTIMATE_KEY_H

#include "compiler/cost_estimator/op_cost_estimate_key.dtg.h"
#include "compiler/cost_estimator/runtime_only_op_cost_estimate_key.dtg.h"
#include "pcg/device_id_t.dtg.h"
#include "pcg/machine_specification.dtg.h"
#include "pcg/parallel_computation_graph/parallel_computation_graph.dtg.h"
#include "pcg/parallel_computation_graph/parallel_layer_guid_t.dtg.h"

namespace FlexFlow {

OpCostEstimateKey get_mapped_op_cost_estimate_key_for_layer(
    ParallelComputationGraph const &pcg,
    OptimizerAttrs const &optimizer_attrs,
    parallel_layer_guid_t const &parallel_layer_guid,
    MachineView const &machine_view);

RuntimeOnlyOpCostEstimateKey
  runtime_only_from_op_cost_estimate_key(OpCostEstimateKey const &);

OpCostEstimateKey
  make_op_cost_estimate_key_from_runtime_only(
    RuntimeOnlyOpCostEstimateKey const &runtime_only,
    OptimizerAttrs const &optimizer_attrs);

} // namespace FlexFlow

#endif
