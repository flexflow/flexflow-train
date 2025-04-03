#include "compiler/mcmc/mcmc_algorithm.h"
#include "compiler/machine_mapping/apply_substitution_and_update_machine_mapping.h"
#include "compiler/machine_mapping/machine_mapping_mutation_set.h"
#include "compiler/mcmc/mcmc_graph_optimize_state.h"
#include "compiler/task_graph_simulator/task_simulator.h"
#include "substitutions/pcg_pattern.h"
#include "substitutions/pcg_pattern_match.h"
#include "substitutions/substitution.h"
#include "substitutions/unity_substitution_set.h"
#include "utils/optional.h"
#include "utils/random_utils.h"

namespace FlexFlow {

bool mcmc_accept(int delta, float temperature) {
  return delta < 0 || randf() < exp(-delta / temperature);
}

void modify_graph_state(MCMCOptimizeState &best_state,
                        MCMCOptimizeState &current_state,
                        SearchResult candidate,
                        CostEstimator const &cost_estimator,
                        MachineSpecification const &resources,
                        MCMCSearchConfig const &search_config) {
  float best_estimate = best_state.runtime;
  float new_estimate = task_simulator_estimate_forward_pass_time(
      candidate.pcg, cost_estimator, candidate.machine_mapping, resources);
  float runtime_delta = new_estimate - best_estimate;
  if (mcmc_accept(runtime_delta, search_config.temperature)) {
    current_state = MCMCOptimizeState{candidate, new_estimate};
    if (runtime_delta < 0) {
      best_state = current_state;
    }
  }
}

SearchResult mcmc_graph_optimize(ParallelComputationGraph &pcg,
                                 CostEstimator const &cost_estimator,
                                 MachineSpecification const &resources,
                                 MCMCSearchConfig const &search_config) {

  std::vector<Substitution> substitutions = get_substitution_set(resources);

  std::optional<MachineMapping> naive_mapping =
      get_naive_mapping(pcg, resources, search_config.device_type);
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

    std::optional<MachineMapping> new_machine_mapping = get_random_mutation(
        current_mapped_pcg, resources, search_config.device_type);
    for (int searched_mutations = 1;
         searched_mutations < search_config.num_mutations_per_iteration;
         searched_mutations++) {
      if (new_machine_mapping == std::nullopt) {
        break;
      }
      modify_graph_state(
          best_state,
          current_state,
          SearchResult{current_mapped_pcg.pcg, new_machine_mapping.value()},
          cost_estimator,
          resources,
          search_config);

      new_machine_mapping = get_random_mutation(
          current_mapped_pcg, resources, search_config.device_type);
    }

    std::optional<Substitution> random_substitution =
        get_random_substitution(resources);
    if (random_substitution != std::nullopt) {
      std::optional<PCGPatternMatch> pattern_match = get_random_pattern_match(
          random_substitution.value().pcg_pattern,
          sub_pcg_from_full_pcg(current_mapped_pcg.pcg));
      if (pattern_match != std::nullopt) {
        SearchResult new_mapped_pcg =
            apply_substitution_and_update_machine_mapping(
                current_mapped_pcg,
                random_substitution.value(),
                pattern_match.value());
        modify_graph_state(best_state,
                           current_state,
                           new_mapped_pcg,
                           cost_estimator,
                           resources,
                           search_config);
      }
    }
  }

  return best_state.mapped_pcg;
}

} // namespace FlexFlow
