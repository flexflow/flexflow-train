#include "compiler/mcmc/mcmc_algorithm.h"
#include "compiler/machine_mapping/machine_mapping_mutation_set.h"
#include "compiler/mcmc/mcmc_graph_optimize_state.h"
#include "compiler/task_graph_simulator/task_simulator.h"
#include "substitutions/apply_substitution/apply_substitution_and_update_machine_mapping.h"
#include "substitutions/pcg_pattern.h"
#include "substitutions/pcg_pattern_match.h"
#include "substitutions/substitution.h"
#include "substitutions/unity_substitution_set.h"
#include "utils/optional.h"
#include "utils/random_utils.h"

namespace FlexFlow {

std::vector<SearchResult> all_pcgs_obtained_by_applying_a_substitution(
    SearchResult const &mapped_pcg,
    std::vector<Substitution> const &substitutions) {
  std::vector<SearchResult> results;
  // currently not functional
  /*SubParallelComputationGraph subpcg = sub_pcg_from_full_pcg(mapped_pcg.pcg);
  for (Substitution const &substitution : substitutions) {
    for (PCGPatternMatch const &pattern_match :
         find_pattern_matches(substitution.pcg_pattern, subpcg)) {
      SearchResult mapped_pcg_from_substitution =
          apply_substitution_and_update_machine_mapping(
              mapped_pcg, substitution, pattern_match);
      results.push_back(mapped_pcg_from_substitution);
    }
  }*/
  return results;
}

bool mcmc_accept(int delta, float temperature) {
  return delta < 0 || randf() < exp(-delta / temperature);
}

SearchResult mcmc_graph_optimize(ParallelComputationGraph &pcg,
                                 CostEstimator const &cost_estimator,
                                 MachineSpecification const &resources,
                                 MCMCSearchConfig const &search_config) {

  std::vector<Substitution> substitutions = get_substitution_set(resources);

  std::optional<MachineMapping> naive_mapping =
      get_naive_mapping(pcg, resources);
  if (naive_mapping == std::nullopt) {
    throw std::runtime_error("Failed to find any solutions");
  }

  MCMCOptimizeState current_state = MCMCOptimizeState{
      SearchResult{pcg, naive_mapping.value()},
      task_simulator_estimate_forward_pass_time(
          pcg, cost_estimator, naive_mapping.value(), resources)};

  MCMCOptimizeState best_state = current_state;

  for (int iteration = 0; iteration < search_config.num_iterations;
       ++iteration) {

    SearchResult current_mapped_pcg = current_state.mapped_pcg;
    float best_estimate = best_state.runtime;

    /*for (SearchResult const &new_mapped_pcg :
         all_pcgs_obtained_by_applying_a_substitution(current_mapped_pcg,
                                                      substitutions)) {
      float new_estimate = task_simulator_estimate_forward_pass_time(
          new_mapped_pcg.pcg,
          cost_estimator,
          new_mapped_pcg.machine_mapping,
          resources);

      if (new_estimate <= search_config.threshold &&
          get_nodes(new_mapped_pcg.pcg.raw_graph).size() <=
              search_config.max_num_ops) {
        candidates.push(MCMCOptimizeState{new_mapped_pcg, -1 * new_estimate});
      }
    }*/

    std::optional<MachineMapping> new_machine_mapping =
        get_random_mutation(current_mapped_pcg, resources);
    for (int searched_mutations = 0;
         searched_mutations < search_config.num_mutations_per_iteration;
         searched_mutations++) {
      if (new_machine_mapping == std::nullopt) {
        break;
      }
      float new_estimate =
          task_simulator_estimate_forward_pass_time(current_mapped_pcg.pcg,
                                                    cost_estimator,
                                                    new_machine_mapping.value(),
                                                    resources);
      float runtime_delta = new_estimate - best_estimate;

      if (mcmc_accept(runtime_delta, search_config.temperature)) {
        current_state = MCMCOptimizeState{
            SearchResult{current_mapped_pcg.pcg, new_machine_mapping.value()},
            new_estimate};
        if (runtime_delta < 0) {
          best_state = current_state;
        }
      }

      new_machine_mapping = get_random_mutation(current_mapped_pcg, resources);
    }
  }

  return best_state.mapped_pcg;
}

} // namespace FlexFlow
