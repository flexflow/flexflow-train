#ifndef _FLEXFLOW_COMPILER_MACHINE_MAPPING_MACHINE_MAPPING_RESULT_H
#define _FLEXFLOW_COMPILER_MACHINE_MAPPING_MACHINE_MAPPING_RESULT_H

#include "compiler/machine_mapping/machine_mapping_result.dtg.h"
#include "compiler/machine_mapping/machine_resource_split.dtg.h"
#include "compiler/machine_mapping/parallel_split_transformation.dtg.h"
#include "utils/units/milliseconds_t.h"

namespace FlexFlow {

[[nodiscard]] MachineMappingResult infeasible_machine_mapping_result();
[[nodiscard]] bool is_infeasible(MachineMappingResult const &);
FeasibleMachineMappingResult require_feasible(MachineMappingResult const &);

[[nodiscard]] MachineMappingResult get_mapping_with_minimal_runtime(
    std::unordered_set<MachineMappingResult> const &);

[[nodiscard]] MachineMappingResult
    series_combine(milliseconds_t comm_cost,
                   MachineMappingResult const &pre_result,
                   MachineMappingResult const &post_result,
                   std::optional<ParallelSplitTransformation> const
                       &parallel_split_transformation);

[[nodiscard]] MachineMappingResult
    parallel_combine(MachineResourceSplit const &split,
                     MachineMappingResult const &lhs_result,
                     MachineMappingResult const &rhs_result);

[[nodiscard]] MachineMappingResult
    minimize_runtime(MachineMappingResult const &m1,
                     MachineMappingResult const &m2);

[[nodiscard]] MachineMappingResult
    make_singleton_machine_mapping_result(milliseconds_t runtime,
                                          MachineView const &machine_view);

[[nodiscard]] float get_runtime_cost(MachineMappingResult const &mm_result);

} // namespace FlexFlow

#endif
