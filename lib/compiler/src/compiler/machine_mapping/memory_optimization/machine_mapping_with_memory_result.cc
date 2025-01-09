#include "compiler/machine_mapping/memory_optimization/machine_mapping_with_memory_result.h"
#include "compiler/machine_mapping/parallel_layer_guid_oblivious_machine_mapping.h"
#include "utils/containers/set_union.h"
#include "utils/full_binary_tree/binary_tree_path.h"

namespace FlexFlow {

MachineMappingWithMemoryResult empty_machine_mapping_with_memory_result() {
  return MachineMappingWithMemoryResult{
      {},
  };
}

MachineMappingWithMemoryResult get_mapping_with_minimal_runtime(
    std::unordered_set<MachineMappingWithMemoryResult> const &candidates) {
  MachineMappingWithMemoryResult result =
      empty_machine_mapping_with_memory_result();

  for (MachineMappingWithMemoryResult const &candidate : candidates) {
    result = minimize_runtime(result, candidate);
  }

  return result;
}

MachineMappingWithMemoryResult remove_non_pareto_optimal_machine_mapping_result(
    MachineMappingWithMemoryResult const &result) {
  std::unordered_set<MachineMappingForSingleLayer> non_pareto_optimal_mappings;
  for (MachineMappingForSingleLayer const &mapping : result.machine_mappings) {
    bool is_pareto_optimal = true;
    for (MachineMappingForSingleLayer const &other_mapping :
         result.machine_mappings) {
      if (mapping.cost.runtime >= other_mapping.cost.runtime &&
          mapping.cost.memory >= other_mapping.cost.memory &&
          mapping != other_mapping) {
        is_pareto_optimal = false;
        break;
      }
    }
    if (is_pareto_optimal) {
      non_pareto_optimal_mappings.insert(mapping);
    }
  }
  return MachineMappingWithMemoryResult{std::move(non_pareto_optimal_mappings)};
}

MachineMappingWithMemoryResult
    series_combine(float comm_cost,
                   MachineMappingWithMemoryResult const &pre_result,
                   MachineMappingWithMemoryResult const &post_result,
                   std::optional<ParallelSplitTransformation> const
                       &parallel_split_transformation) {
  auto combine_machine_mapping =
      [&](MachineMappingForSingleLayer const &pre_mm,
          MachineMappingForSingleLayer const &post_mm) {
        OpCostMetrics cost = OpCostMetrics{
            pre_mm.cost.runtime + comm_cost + post_mm.cost.runtime,
            pre_mm.cost.memory + post_mm.cost.memory,
        };

        ParallelLayerGuidObliviousMachineMapping mapping = [&] {
          if (parallel_split_transformation.has_value() &&
              parallel_split_transformation.value() ==
                  ParallelSplitTransformation::RthenL) {
            return binary_combine_mappings(/*lhs=*/post_mm.machine_mapping,
                                           /*rhs=*/pre_mm.machine_mapping);
          } else {
            return binary_combine_mappings(/*lhs=*/pre_mm.machine_mapping,
                                           /*rhs=*/post_mm.machine_mapping);
          }
        }();

        return MachineMappingForSingleLayer{cost, mapping};
      };

  MachineMappingWithMemoryResult result =
      empty_machine_mapping_with_memory_result();
  for (MachineMappingForSingleLayer const &pre_mm :
       pre_result.machine_mappings) {
    for (MachineMappingForSingleLayer const &post_mm :
         post_result.machine_mappings) {
      result.machine_mappings.insert(combine_machine_mapping(pre_mm, post_mm));
    }
  }

  return remove_non_pareto_optimal_machine_mapping_result(result);
}

MachineMappingWithMemoryResult
    parallel_combine(MachineMappingWithMemoryResult const &lhs_result,
                     MachineMappingWithMemoryResult const &rhs_result) {
  auto combine_machine_mapping =
      [&](MachineMappingForSingleLayer const &lhs_mm,
          MachineMappingForSingleLayer const &rhs_mm) {
        OpCostMetrics cost = OpCostMetrics{
            std::max(lhs_mm.cost.runtime, rhs_mm.cost.runtime),
            std::max(lhs_mm.cost.memory, rhs_mm.cost.memory),
        };

        ParallelLayerGuidObliviousMachineMapping mapping =
            binary_combine_mappings(lhs_mm.machine_mapping,
                                    rhs_mm.machine_mapping);

        return MachineMappingForSingleLayer{cost, mapping};
      };

  MachineMappingWithMemoryResult result =
      empty_machine_mapping_with_memory_result();
  for (MachineMappingForSingleLayer const &lhs_mm :
       lhs_result.machine_mappings) {
    for (MachineMappingForSingleLayer const &rhs_mm :
         rhs_result.machine_mappings) {
      result.machine_mappings.insert(combine_machine_mapping(lhs_mm, rhs_mm));
    }
  }

  return remove_non_pareto_optimal_machine_mapping_result(result);
}

MachineMappingWithMemoryResult
    minimize_runtime(MachineMappingWithMemoryResult const &m1,
                     MachineMappingWithMemoryResult const &m2) {
  MachineMappingWithMemoryResult result = MachineMappingWithMemoryResult{
      set_union(m1.machine_mappings, m2.machine_mappings),
  };
  return remove_non_pareto_optimal_machine_mapping_result(result);
}

MachineMappingWithMemoryResult
    make_singleton_machine_mapping_with_memory_result(
        OpCostMetrics cost, MachineView const &machine_view) {
  return MachineMappingWithMemoryResult{{
      MachineMappingForSingleLayer{
          cost,
          ParallelLayerGuidObliviousMachineMapping{{
              {binary_tree_root_path(), machine_view},
          }},
      },
  }};
}

} // namespace FlexFlow
