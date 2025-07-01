#ifndef _FLEXFLOW_LIB_COMPILER_INCLUDE_COMPILER_MACHINE_MAPPING_MACHINE_MAPPING_PROBLEM_TREE_UNMAPPED_OP_COST_ESTIMATE_KEY_H
#define _FLEXFLOW_LIB_COMPILER_INCLUDE_COMPILER_MACHINE_MAPPING_MACHINE_MAPPING_PROBLEM_TREE_UNMAPPED_OP_COST_ESTIMATE_KEY_H

#include "compiler/cost_estimator/op_cost_estimate_key.dtg.h"
#include "compiler/machine_mapping/machine_mapping_problem_tree/unmapped_op_cost_estimate_key.dtg.h"
#include "compiler/machine_mapping/machine_mapping_problem_tree/unmapped_runtime_only_op_cost_estimate_key.dtg.h"
#include "pcg/parallel_computation_graph/parallel_computation_graph.dtg.h"
#include "pcg/parallel_computation_graph/parallel_layer_guid_t.dtg.h"

namespace FlexFlow {

UnmappedOpCostEstimateKey get_unmapped_op_cost_estimate_key_for_layer(
    ParallelComputationGraph const &pcg, 
    OptimizerAttrs const &optimizer_attrs,
    parallel_layer_guid_t const &parallel_layer_guid);

UnmappedOpCostEstimateKey 
  unmapped_op_cost_estimate_key_from_runtime_only(
    UnmappedRuntimeOnlyOpCostEstimateKey const &runtime_only,
    OptimizerAttrs const &optimizer_attrs); 

UnmappedRuntimeOnlyOpCostEstimateKey
  runtime_only_from_unmapped_op_cost_estimate_key(
    UnmappedOpCostEstimateKey const &runtime_only);

OpCostEstimateKey
    map_unmapped_op_cost_estimate_key(UnmappedOpCostEstimateKey const &unmapped,
                                      MachineView const &machine_view);

} // namespace FlexFlow

#endif
