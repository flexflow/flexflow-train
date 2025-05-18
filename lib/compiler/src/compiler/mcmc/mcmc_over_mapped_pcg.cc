#include "compiler/mcmc/mcmc_over_mapped_pcg.h"
#include "compiler/machine_mapping/apply_substitution_and_update_machine_mapping.h"
#include "compiler/machine_mapping/machine_mapping_mutation_set.h"
#include "compiler/mcmc/generic_mcmc_algorithm.h"
#include "compiler/task_graph_simulator/task_simulator.h"
#include "substitutions/pcg_pattern.h"
#include "substitutions/pcg_pattern_match.h"
#include "substitutions/unity_substitution_set.h"
#include "compiler/search_result.h"
#include "utils/optional.h"

namespace FlexFlow {

SearchResult mcmc_graph_optimize(ParallelComputationGraph &pcg,
                                 CostEstimator const &cost_estimator,
                                 MachineSpecification const &resources,
                                 MCMCOverMappedPCGConfig const &search_config) {

  std::vector<Substitution> substitutions = get_substitution_set(resources);

  std::optional<MachineMapping> naive_mapping =
      get_naive_mapping(pcg, resources, search_config.device_type);
  if (naive_mapping == std::nullopt) {
    throw std::runtime_error("Failed to find any solutions");
  }

  SearchResult starting_state = SearchResult{pcg, naive_mapping.value()};

  auto generating_func = [&](SearchResult mapped_pcg,
                             nonnegative_int i) -> std::optional<SearchResult> {
    if (i.unwrap_nonnegative() %
            search_config.substitution_interval.unwrap_nonnegative() ==
        0) {
      // substitutions every (substitution_interval) iterations
      std::optional<Substitution> random_substitution =
          get_random_substitution(resources);
      if (random_substitution != std::nullopt) {
        std::optional<PCGPatternMatch> pattern_match =
            get_random_pattern_match(random_substitution.value().pcg_pattern,
                                     sub_pcg_from_full_pcg(mapped_pcg.pcg));
        if (pattern_match != std::nullopt) {
          return apply_substitution_and_update_machine_mapping(
              mapped_pcg, random_substitution.value(), pattern_match.value());
        }
      }
      return std::nullopt;
    } else {
      // machine mapping mutations otherwise
      std::optional<MachineMapping> new_machine_mapping =
          get_random_mutation(mapped_pcg, resources, search_config.device_type);
      if (new_machine_mapping == std::nullopt) {
        return std::nullopt;
      }
      return SearchResult{mapped_pcg.pcg, new_machine_mapping.value()};
    }
  };

  auto scoring_func = [&](SearchResult mapped_pcg) -> float {
    return task_simulator_estimate_forward_pass_time(
        mapped_pcg.pcg, cost_estimator, mapped_pcg.machine_mapping, resources);
  };

  GenericMCMCConfig config =
      GenericMCMCConfig{/*temperature*/ search_config.temperature,
                        /*num_iterations*/ search_config.num_iterations};

  Generic_MCMC_state<SearchResult, float> result =
      minimize_score(starting_state, generating_func, scoring_func, config);

  return result.get_state();
}

} // namespace FlexFlow
