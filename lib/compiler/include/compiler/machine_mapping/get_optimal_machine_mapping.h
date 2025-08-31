#ifndef _FLEXFLOW_COMPILER_MACHINE_MAPPING_GET_OPTIMAL_MACHINE_MAPPING_H
#define _FLEXFLOW_COMPILER_MACHINE_MAPPING_GET_OPTIMAL_MACHINE_MAPPING_H

#include "compiler/machine_mapping/machine_compute_resource_slice.dtg.h"
#include "compiler/machine_mapping/machine_mapping_cache.dtg.h"
#include "compiler/machine_mapping/machine_mapping_constraints.dtg.h"
#include "compiler/machine_mapping/machine_mapping_context.dtg.h"
#include "compiler/machine_mapping/machine_mapping_problem_tree/machine_mapping_problem_tree.dtg.h"
#include "compiler/machine_mapping/machine_mapping_problem_tree/mm_problem_tree_parallel_split.dtg.h"
#include "compiler/machine_mapping/machine_mapping_problem_tree/mm_problem_tree_series_split.dtg.h"
#include "compiler/machine_mapping/parallel_split_transformation.dtg.h"

namespace FlexFlow {

MachineMappingResult
    get_optimal_machine_mapping(MachineMappingCache &result_cache,
                                MachineMappingContext const &context,
                                MachineMappingProblemTree const &problem_tree,
                                MachineComputeResourceSlice const &resources,
                                MachineMappingConstraints const &constraints);

MachineMappingResult
    get_optimal_machine_mapping(MachineMappingCache &result_cache,
                                MachineMappingContext const &context,
                                MMProblemTreeSeriesSplit const &series_split,
                                MachineComputeResourceSlice const &resources,
                                MachineMappingConstraints const &constraints,
                                std::optional<ParallelSplitTransformation> const
                                    &parallel_split_transformation);

MachineMappingResult get_optimal_machine_mapping(
    MachineMappingCache &result_cache,
    MachineMappingContext const &context,
    MMProblemTreeParallelSplit const &parallel_split,
    MachineComputeResourceSlice const &resources,
    MachineMappingConstraints const &constraints);

MachineMappingResult get_optimal_machine_mapping(
    MachineMappingCache &result_cache,
    MachineMappingContext const &,
    UnmappedRuntimeOnlyOpCostEstimateKey const &leaf,
    MachineComputeResourceSlice const &resources,
    MachineMappingConstraints const &constraints);

} // namespace FlexFlow

#endif
