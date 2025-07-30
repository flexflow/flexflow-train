#ifndef _FLEXFLOW_COMPILER_MACHINE_MAPPING_MEMORY_OPTIMIZATION_GET_OPTIMAL_MACHINE_MAPPING_WITH_MEMORY_H
#define _FLEXFLOW_COMPILER_MACHINE_MAPPING_MEMORY_OPTIMIZATION_GET_OPTIMAL_MACHINE_MAPPING_WITH_MEMORY_H

#include "compiler/machine_mapping/machine_mapping_cache.dtg.h"
#include "compiler/machine_mapping/machine_mapping_constraints.dtg.h"
#include "compiler/machine_mapping/machine_mapping_problem_tree/machine_mapping_problem_tree.dtg.h"
#include "compiler/machine_mapping/machine_mapping_problem_tree/mm_problem_tree_parallel_split.dtg.h"
#include "compiler/machine_mapping/machine_mapping_problem_tree/mm_problem_tree_series_split.dtg.h"
#include "compiler/machine_mapping/machine_mapping_problem_tree/unmapped_op_cost_estimate_key.dtg.h"
#include "compiler/machine_mapping/memory_optimization/machine_mapping_with_memory_cache.dtg.h"
#include "compiler/machine_mapping/memory_optimization/machine_mapping_with_memory_context.dtg.h"
#include "compiler/machine_mapping/parallel_split_transformation.dtg.h"
#include "pcg/machine_compute_specification.dtg.h"

namespace FlexFlow {

MachineMappingWithMemoryResult get_optimal_machine_mapping_with_memory(
    MachineMappingWithMemoryCache &result_cache,
    MachineMappingWithMemoryContext const &context,
    MachineMappingProblemTree const &problem_tree,
    MachineComputeSpecification const &resources,
    MachineMappingConstraints const &constraints);

MachineMappingWithMemoryResult get_optimal_machine_mapping_with_memory(
    MachineMappingWithMemoryCache &result_cache,
    MachineMappingWithMemoryContext const &context,
    MMProblemTreeSeriesSplit const &series_split,
    MachineComputeSpecification const &resources,
    MachineMappingConstraints const &constraints,
    std::optional<ParallelSplitTransformation> const
        &parallel_split_transformation);

MachineMappingWithMemoryResult get_optimal_machine_mapping_with_memory(
    MachineMappingWithMemoryCache &result_cache,
    MachineMappingWithMemoryContext const &context,
    MMProblemTreeParallelSplit const &parallel_split,
    MachineComputeSpecification const &resources,
    MachineMappingConstraints const &constraints);

MachineMappingWithMemoryResult get_optimal_machine_mapping_with_memory(
    MachineMappingWithMemoryCache &result_cache,
    MachineMappingWithMemoryContext const &context,
    UnmappedRuntimeOnlyOpCostEstimateKey const &leaf,
    MachineComputeSpecification const &resources,
    MachineMappingConstraints const &constraints);

} // namespace FlexFlow

#endif
