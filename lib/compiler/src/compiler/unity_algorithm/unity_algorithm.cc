#include "compiler/unity_algorithm/unity_algorithm.h"
#include "compiler/machine_mapping/get_optimal_machine_mapping.h"
#include "compiler/machine_mapping/machine_mapping.h"
#include "compiler/machine_mapping/machine_mapping_problem_tree/get_machine_mapping_problem_tree.h"
#include "compiler/machine_mapping/machine_mapping_problem_tree/unmapped_op_cost_estimate_key.h"
#include "compiler/series_parallel/pcg/get_pcg_balanced_binary_sp_decomposition.h"
#include "compiler/unity_algorithm/allowed_machine_views.h"
#include "compiler/unity_algorithm/graph_optimize_state.h"
#include "pcg/machine_specification.dtg.h"
#include "pcg/operator_task_space.h"
#include "substitutions/pcg_pattern.h"
#include "substitutions/sub_parallel_computation_graph.h"
#include "substitutions/substitution.h"
#include "utils/deduplicated_priority_queue.h"
#include "utils/graph/node/algorithms.h"
#include "compiler/machine_mapping/machine_mapping_cache.h"
#include "compiler/machine_mapping/machine_mapping_constraints.h"
#include "compiler/machine_mapping/machine_mapping_problem_tree/machine_mapping_problem_tree.h"

namespace FlexFlow {

/*
 * Applies a substitution to all possible positions in PCG
 */
std::vector<ParallelComputationGraph>
    all_pcgs_obtained_by_applying_a_substitution(
        ParallelComputationGraph const &pcg,
        std::vector<Substitution> const &substitutions) {
  std::vector<ParallelComputationGraph> results;
  SubParallelComputationGraph subpcg = sub_pcg_from_full_pcg(pcg);
  for (Substitution const &substitution : substitutions) {
    for (PCGPatternMatch const &pattern_match :
         find_pattern_matches(substitution.pcg_pattern, subpcg)) {
      SubParallelComputationGraph subpcg_from_substitution =
          apply_substitution(subpcg, substitution, pattern_match);
      results.push_back(
          pcg_from_sub_pcg_by_dropping_inputs(subpcg_from_substitution));
    }
  }
  return results;
}

SearchResult graph_optimize(ParallelComputationGraph &pcg,
                            CostEstimator const &cost_estimator,
                            MachineSpecification const &resources,
                            std::vector<Substitution> const &substitutions,
                            UnitySearchConfig const &search_config,
                            DeviceType device_type) {

  // NOTE(@wmdi): This mapping is only used for allowed_machine_views
  std::unordered_map<UnmappedOpCostEstimateKey, parallel_layer_guid_t>
      mapping_from_unmapped_op_cost_estimate_key_parallel_layer = [&] {
        std::unordered_map<UnmappedOpCostEstimateKey, parallel_layer_guid_t>
            mapping;
        for (parallel_layer_guid_t layer : get_parallel_layers(pcg)) {
          // NOTE(@wmdi): Assume layers with the same key have the same allowed
          // machine views
          mapping.insert(
              {get_unmapped_op_cost_estimate_key_for_layer(pcg, layer), layer});
        }
        return mapping;
      }();

  MachineMappingCache cached_subgraph_costs = empty_machine_mapping_cache();
  DeduplicatedPriorityQueue<GraphOptimizeState> candidates;

  MachineMappingContext context = MachineMappingContext{
      /*cost_estimator=*/cost_estimator,
      /*allowed_machine_views=*/
      [&](UnmappedOpCostEstimateKey const &key,
          MachineSpecification const &resources)
          -> std::unordered_set<MachineView> {
        return get_allowed_machine_views(
            resources,
            get_operator_task_space(
                pcg,
                mapping_from_unmapped_op_cost_estimate_key_parallel_layer.at(
                    key)),
            device_type);
      },
  };

  auto get_runtime_cost = [](MachineMappingResult const &mm_result) {
    if (mm_result.raw_result == std::nullopt) {
      return std::numeric_limits<float>::infinity();
    } else {
      return mm_result.raw_result.value().runtime;
    }
  };

  auto optimize_pcg = [&](ParallelComputationGraph const &pcg)
      -> std::pair<GraphOptimizeState, MachineMapping> {
    std::optional<PCGBinarySPDecomposition> maybe_sp_decomp =
        get_pcg_balanced_binary_sp_decomposition(pcg);

    if (!maybe_sp_decomp.has_value()) {
      throw std::runtime_error("Fail to SP-ize PCG");
    }

    PCGBinarySPDecomposition sp_decomp = maybe_sp_decomp.value();

    MachineMappingProblemTree problem_tree = get_machine_mapping_problem_tree(pcg, sp_decomp);
    MachineMappingConstraints constraints = 
      get_unconstrained_solution_for_layers(get_all_leaf_paths(problem_tree));

    MachineMappingResult mm_result = get_optimal_machine_mapping(
        cached_subgraph_costs,
        context,
        get_machine_mapping_problem_tree(pcg, sp_decomp),
        resources,
        constraints);

    return {
        GraphOptimizeState{
            /*pcg=*/pcg,
            /*runtime_with_optimal_mm=*/get_runtime_cost(mm_result),
        },
        get_machine_mapping_from_machine_mapping_result(sp_decomp, mm_result),
    };
  };

  GraphOptimizeState best_state = optimize_pcg(pcg).first;
  candidates.push(best_state);

  for (int iteration = 0;
       !candidates.empty() && iteration < search_config.budget;
       ++iteration) {
    GraphOptimizeState current_state = candidates.top();
    candidates.pop();

    if (current_state < best_state) {
      best_state = current_state;
    } else if (current_state.runtime_with_optimal_mm >
               best_state.runtime_with_optimal_mm * search_config.alpha) {
      continue;
    }

    for (ParallelComputationGraph const &new_pcg :
         all_pcgs_obtained_by_applying_a_substitution(current_state.pcg,
                                                      substitutions)) {
      std::optional<GraphOptimizeState> new_pcg_optimize_result =
          optimize_pcg(new_pcg).first;
      if (new_pcg_optimize_result == std::nullopt) {
        continue;
      }
      GraphOptimizeState new_state = new_pcg_optimize_result.value();
      if (new_state.runtime_with_optimal_mm <= search_config.threshold &&
          get_nodes(new_pcg.raw_graph).size() <= search_config.max_num_ops) {
        candidates.push(new_state);
      }
    }
  }

  return SearchResult{
      /*pcg=*/best_state.pcg,
      /*machine_mapping=*/optimize_pcg(best_state.pcg).second,
  };
}

} // namespace FlexFlow
