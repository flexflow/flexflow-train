#include "compiler/machine_mapping/memory_optimization/machine_mapping_with_memory_result.h"
#include "compiler/machine_mapping/machine_resource_split.h"
#include "compiler/machine_mapping/parallel_layer_guid_oblivious_machine_mapping.h"
#include "utils/containers/set_union.h"
#include "utils/full_binary_tree/binary_tree_path.h"
#include "utils/containers/all_of.h"
#include "utils/containers/transform.h"
#include "compiler/machine_mapping/memory_optimization/pareto_optimal_machine_mapping.h"
#include "utils/hash/tuple.h"
#include "utils/hash/unordered_set.h"

namespace FlexFlow {

MachineMappingWithMemoryResult::MachineMappingWithMemoryResult(
  std::unordered_set<ParetoOptimalMachineMapping> const &pareto_frontier) 
  : m_pareto_frontier(pareto_frontier)
{
  ASSERT(
    all_of(pareto_frontier,
           [&](ParetoOptimalMachineMapping const &m) {
             return is_pareto_optimal_in(m, pareto_frontier);
           }));
}

bool MachineMappingWithMemoryResult::operator==(MachineMappingWithMemoryResult const &other) const {
  return this->tie() == other.tie();
}

bool MachineMappingWithMemoryResult::operator!=(MachineMappingWithMemoryResult const &other) const {
  return this->tie() != other.tie();
}

std::unordered_set<ParetoOptimalMachineMapping> const &MachineMappingWithMemoryResult::get_pareto_frontier() const {
  return this->m_pareto_frontier;
}

std::string format_as(MachineMappingWithMemoryResult const &r) {
  return fmt::format("<MachineMappingWithMemoryResult pareto_frontier={}>", r.get_pareto_frontier());
}

std::ostream &operator<<(std::ostream &s, MachineMappingWithMemoryResult const &r) {
  return (s << fmt::to_string(r));
}

std::tuple<
  std::unordered_set<ParetoOptimalMachineMapping> const &
> MachineMappingWithMemoryResult::tie() const {
  return std::tie(this->m_pareto_frontier);
}

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

MachineMappingWithMemoryResult
    series_combine(milliseconds_t comm_cost,
                   MachineMappingWithMemoryResult const &pre_result,
                   MachineMappingWithMemoryResult const &post_result,
                   std::optional<ParallelSplitTransformation> const
                       &parallel_split_transformation) {
  auto combine_machine_mapping =
      [&](ParetoOptimalMachineMapping const &pre_mm,
          ParetoOptimalMachineMapping const &post_mm) {
        OpCostMetrics cost = OpCostMetrics{
            /*forward_runtime=*/pre_mm.cost.forward_runtime + comm_cost +
                post_mm.cost.forward_runtime,
            /*backward_runtime=*/pre_mm.cost.backward_runtime + comm_cost +
                post_mm.cost.backward_runtime,
            /*memory_usage=*/pre_mm.cost.memory_usage +
                post_mm.cost.memory_usage,
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

        return ParetoOptimalMachineMapping{cost, mapping};
      };

  std::unordered_set<ParetoOptimalMachineMapping> result;

  for (ParetoOptimalMachineMapping const &pre_mm :
       pre_result.get_pareto_frontier()) {
    for (ParetoOptimalMachineMapping const &post_mm :
         post_result.get_pareto_frontier()) {
      result.insert(combine_machine_mapping(pre_mm, post_mm));
    }
  }

  return MachineMappingWithMemoryResult{
    /*pareto_frontier=*/filter(result,
           [&](ParetoOptimalMachineMapping const &m) {
             return is_pareto_optimal_in(m, result);
           }),
  }; 
}

MachineMappingWithMemoryResult
    parallel_combine(MachineResourceSplit const &split,
                     MachineMappingWithMemoryResult const &lhs_result,
                     MachineMappingWithMemoryResult const &rhs_result) {
  auto combine_machine_mapping =
      [&](ParetoOptimalMachineMapping const &lhs_mm,
          ParetoOptimalMachineMapping const &rhs_mm) {
        OpCostMetrics cost = OpCostMetrics{
            /*forward_runtime=*/
            std::max(lhs_mm.cost.forward_runtime, rhs_mm.cost.forward_runtime),
            /*backward_runtime=*/
            std::max(lhs_mm.cost.backward_runtime,
                     rhs_mm.cost.backward_runtime),
            /*memory_usage=*/
            std::max(lhs_mm.cost.memory_usage, rhs_mm.cost.memory_usage),
        };

        ParallelLayerGuidObliviousMachineMapping mapping =
            binary_combine_mappings(
              lhs_mm.machine_mapping,
              offset_layer_oblivious_mapping_by(rhs_mm.machine_mapping, split));

        return ParetoOptimalMachineMapping{cost, mapping};
      };

  std::unordered_set<ParetoOptimalMachineMapping> result;
  for (ParetoOptimalMachineMapping const &lhs_mm :
       lhs_result.get_pareto_frontier()) {

    for (ParetoOptimalMachineMapping const &rhs_mm :
         rhs_result.get_pareto_frontier()) {

      result.insert(combine_machine_mapping(lhs_mm, rhs_mm));
    }
  }

  return MachineMappingWithMemoryResult{
    /*pareto_frontier=*/filter(result,
           [&](ParetoOptimalMachineMapping const &m) {
             return is_pareto_optimal_in(m, result);
           }),
  }; 
}

MachineMappingWithMemoryResult
    minimize_runtime(MachineMappingWithMemoryResult const &m1,
                     MachineMappingWithMemoryResult const &m2) {
  std::unordered_set<ParetoOptimalMachineMapping> result = 
      set_union(m1.get_pareto_frontier(), m2.get_pareto_frontier());

  return MachineMappingWithMemoryResult{
    /*pareto_frontier=*/filter(result,
           [&](ParetoOptimalMachineMapping const &m) {
             return is_pareto_optimal_in(m, result);
           }),
  }; 
}

MachineMappingWithMemoryResult
    make_singleton_machine_mapping_with_memory_result(
        OpCostMetrics cost, MachineView const &machine_view) {
  return MachineMappingWithMemoryResult{{
      ParetoOptimalMachineMapping{
          cost,
          ParallelLayerGuidObliviousMachineMapping{{
              {binary_tree_root_path(), machine_view},
          }},
      },
  }};
}

} // namespace FlexFlow

namespace std {

size_t hash<::FlexFlow::MachineMappingWithMemoryResult>::operator()(::FlexFlow::MachineMappingWithMemoryResult const &r) const {
  return get_std_hash(r.tie());
}

}
