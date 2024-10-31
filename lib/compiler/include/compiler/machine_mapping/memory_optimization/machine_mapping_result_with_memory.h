#ifndef _FLEXFLOW_COMPILER_MACHINE_MAPPING_MEMORY_OPTIMIZATION_MACHINE_MAPPING_RESULT_WITH_MEMORY_H
#define _FLEXFLOW_COMPILER_MACHINE_MAPPING_MEMORY_OPTIMIZATION_MACHINE_MAPPING_RESULT_WITH_MEMORY_H

#include "compiler/machine_mapping/memory_optimization/machine_mapping_result_with_memory.dtg.h"
#include "compiler/machine_mapping/parallel_split_transformation.dtg.h"
#include <optional>

namespace FlexFlow {

[[nodiscard]] MachineMappingResultWithMemory
    empty_machine_mapping_result_with_memory();
[[nodiscard]] bool is_empty(MachineMappingResultWithMemory const &);

[[nodiscard]] MachineMappingResultWithMemory get_mapping_with_minimal_runtime(
    std::unordered_set<MachineMappingResultWithMemory> const &);

[[nodiscard]] MachineMappingResultWithMemory
    remove_non_dominating_machine_mapping_result(
        MachineMappingResultWithMemory const &);

[[nodiscard]] MachineMappingResultWithMemory
    series_combine(float comm_cost,
                   MachineMappingResultWithMemory const &pre_result,
                   MachineMappingResultWithMemory const &post_result,
                   std::optional<ParallelSplitTransformation> const
                       &parallel_split_transformation);
[[nodiscard]] MachineMappingResultWithMemory
    parallel_combine(MachineMappingResultWithMemory const &lhs_result,
                     MachineMappingResultWithMemory const &rhs_result);

[[nodiscard]] MachineMappingResultWithMemory
    minimize_runtime(MachineMappingResultWithMemory const &m1,
                     MachineMappingResultWithMemory const &m2);

[[nodiscard]] MachineMappingResultWithMemory
    make_singleton_machine_mapping_result_with_memory(
        CostMetric cost, MachineView const &machine_view);

} // namespace FlexFlow

#endif
