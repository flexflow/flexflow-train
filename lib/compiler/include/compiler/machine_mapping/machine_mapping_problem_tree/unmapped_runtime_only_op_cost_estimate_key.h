#ifndef _FLEXFLOW_LIB_COMPILER_INCLUDE_COMPILER_MACHINE_MAPPING_MACHINE_MAPPING_PROBLEM_TREE_UNMAPPED_RUNTIME_ONLY_OP_COST_ESTIMATE_KEY_H
#define _FLEXFLOW_LIB_COMPILER_INCLUDE_COMPILER_MACHINE_MAPPING_MACHINE_MAPPING_PROBLEM_TREE_UNMAPPED_RUNTIME_ONLY_OP_COST_ESTIMATE_KEY_H

#include "compiler/cost_estimator/runtime_only_op_cost_estimate_key.dtg.h"
#include "compiler/machine_mapping/machine_mapping_problem_tree/unmapped_runtime_only_op_cost_estimate_key.dtg.h"
#include "pcg/parallel_computation_graph/parallel_layer_guid_t.dtg.h"
#include "pcg/parallel_computation_graph/parallel_computation_graph.dtg.h"
#include "pcg/parallel_computation_graph/parallel_layer_guid_t.dtg.h"

namespace FlexFlow {

UnmappedRuntimeOnlyOpCostEstimateKey get_unmapped_runtime_only_op_cost_estimate_key_for_layer(
    ParallelComputationGraph const &pcg, 
    parallel_layer_guid_t const &parallel_layer_guid);

RuntimeOnlyOpCostEstimateKey
    map_unmapped_runtime_only_op_cost_estimate_key(UnmappedRuntimeOnlyOpCostEstimateKey const &unmapped,
                                                   MachineView const &machine_view);

} // namespace FlexFlow

#endif
