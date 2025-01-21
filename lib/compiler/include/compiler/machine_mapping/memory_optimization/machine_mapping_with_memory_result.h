#ifndef _FLEXFLOW_COMPILER_MACHINE_MAPPING_MEMORY_OPTIMIZATION_MACHINE_MAPPING_RESULT_WITH_MEMORY_H
#define _FLEXFLOW_COMPILER_MACHINE_MAPPING_MEMORY_OPTIMIZATION_MACHINE_MAPPING_RESULT_WITH_MEMORY_H

#include "compiler/machine_mapping/memory_optimization/machine_mapping_with_memory_result.dtg.h"
#include "compiler/machine_mapping/parallel_split_transformation.dtg.h"
#include <optional>

namespace FlexFlow {

[[nodiscard]] MachineMappingWithMemoryResult
    empty_machine_mapping_with_memory_result();
[[nodiscard]] bool is_empty(MachineMappingWithMemoryResult const &);

[[nodiscard]] MachineMappingWithMemoryResult get_mapping_with_minimal_runtime(
    std::unordered_set<MachineMappingWithMemoryResult> const &);

[[nodiscard]] MachineMappingWithMemoryResult
    remove_non_pareto_optimal_machine_mapping_result(
        MachineMappingWithMemoryResult const &);

[[nodiscard]] MachineMappingWithMemoryResult
    series_combine(float comm_cost,
                   MachineMappingWithMemoryResult const &pre_result,
                   MachineMappingWithMemoryResult const &post_result,
                   std::optional<ParallelSplitTransformation> const
                       &parallel_split_transformation);
[[nodiscard]] MachineMappingWithMemoryResult
    parallel_combine(MachineMappingWithMemoryResult const &lhs_result,
                     MachineMappingWithMemoryResult const &rhs_result);

[[nodiscard]] MachineMappingWithMemoryResult
    minimize_runtime(MachineMappingWithMemoryResult const &m1,
                     MachineMappingWithMemoryResult const &m2);

[[nodiscard]] MachineMappingWithMemoryResult
    make_singleton_machine_mapping_with_memory_result(
        OpCostMetrics cost, MachineView const &machine_view);

} // namespace FlexFlow

#endif
