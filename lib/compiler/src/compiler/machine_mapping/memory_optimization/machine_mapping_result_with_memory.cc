#include "compiler/machine_mapping/memory_optimization/machine_mapping_result_with_memory.h"
#include "compiler/machine_mapping/parallel_layer_guid_oblivious_machine_mapping.h"
#include "utils/containers/set_union.h"
#include "utils/full_binary_tree/binary_tree_path.h"

namespace FlexFlow {

MachineMappingResultWithMemory empty_machine_mapping_result_with_memory() {
  return MachineMappingResultWithMemory{
      {},
  };
}

MachineMappingResultWithMemory get_mapping_with_minimal_runtime(
    std::unordered_set<MachineMappingResultWithMemory> const &candidates) {
  MachineMappingResultWithMemory result =
      empty_machine_mapping_result_with_memory();

  for (MachineMappingResultWithMemory const &candidate : candidates) {
    result = minimize_runtime(result, candidate);
  }

  return result;
}

MachineMappingResultWithMemory remove_non_dominating_machine_mapping_result(
    MachineMappingResultWithMemory const &result) {
  std::unordered_set<SingleMachineMapping> non_dominating_mappings;
  for (SingleMachineMapping const &mapping : result.machine_mappings) {
    bool is_dominating = true;
    for (SingleMachineMapping const &other_mapping : result.machine_mappings) {
      if (mapping.cost.runtime >= other_mapping.cost.runtime &&
          mapping.cost.memory >= other_mapping.cost.memory &&
          mapping != other_mapping) {
        is_dominating = false;
        break;
      }
    }
    if (is_dominating) {
      non_dominating_mappings.insert(mapping);
    }
  }
  return MachineMappingResultWithMemory{std::move(non_dominating_mappings)};
}

MachineMappingResultWithMemory
    series_combine(float comm_cost,
                   MachineMappingResultWithMemory const &pre_result,
                   MachineMappingResultWithMemory const &post_result,
                   std::optional<ParallelSplitTransformation> const
                       &parallel_split_transformation) {
  auto combine_machine_mapping = [&](SingleMachineMapping const &pre_mm,
                                     SingleMachineMapping const &post_mm) {
    CostMetric cost = CostMetric{
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

    return SingleMachineMapping{cost, mapping};
  };

  MachineMappingResultWithMemory result =
      empty_machine_mapping_result_with_memory();
  for (SingleMachineMapping const &pre_mm : pre_result.machine_mappings) {
    for (SingleMachineMapping const &post_mm : post_result.machine_mappings) {
      result.machine_mappings.insert(combine_machine_mapping(pre_mm, post_mm));
    }
  }

  return remove_non_dominating_machine_mapping_result(result);
}

MachineMappingResultWithMemory
    parallel_combine(MachineMappingResultWithMemory const &lhs_result,
                     MachineMappingResultWithMemory const &rhs_result) {
  auto combine_machine_mapping = [&](SingleMachineMapping const &lhs_mm,
                                     SingleMachineMapping const &rhs_mm) {
    CostMetric cost = CostMetric{
        std::max(lhs_mm.cost.runtime, rhs_mm.cost.runtime),
        std::max(lhs_mm.cost.memory, rhs_mm.cost.memory),
    };

    ParallelLayerGuidObliviousMachineMapping mapping =
        binary_combine_mappings(lhs_mm.machine_mapping, rhs_mm.machine_mapping);

    return SingleMachineMapping{cost, mapping};
  };

  MachineMappingResultWithMemory result =
      empty_machine_mapping_result_with_memory();
  for (SingleMachineMapping const &lhs_mm : lhs_result.machine_mappings) {
    for (SingleMachineMapping const &rhs_mm : rhs_result.machine_mappings) {
      result.machine_mappings.insert(combine_machine_mapping(lhs_mm, rhs_mm));
    }
  }

  return remove_non_dominating_machine_mapping_result(result);
}

MachineMappingResultWithMemory
    minimize_runtime(MachineMappingResultWithMemory const &m1,
                     MachineMappingResultWithMemory const &m2) {
  MachineMappingResultWithMemory result = MachineMappingResultWithMemory{
      set_union(m1.machine_mappings, m2.machine_mappings),
  };
  return remove_non_dominating_machine_mapping_result(result);
}

MachineMappingResultWithMemory
    make_singleton_machine_mapping_result_with_memory(
        CostMetric cost, MachineView const &machine_view) {
  return MachineMappingResultWithMemory{{
      SingleMachineMapping{
          cost,
          ParallelLayerGuidObliviousMachineMapping{{
              {binary_tree_root_path(), machine_view},
          }},
      },
  }};
}

} // namespace FlexFlow
