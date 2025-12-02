#include "compiler/mcmc/mcmc_over_mapped_pcg.h"
#include "compiler/machine_mapping/apply_substitution_and_update_machine_mapping.h"
#include "compiler/machine_mapping/machine_mapping_mutation_set.h"
#include "compiler/mcmc/generic_mcmc_algorithm.h"
#include "compiler/search_result.h"
#include "compiler/task_graph_simulator/task_simulator.h"
#include "substitutions/pcg_pattern.h"
#include "substitutions/pcg_pattern_match.h"
#include "substitutions/unity_substitution_set.h"
#include "utils/optional.h"
#include "utils/random_utils.h"
#include <libassert/assert.hpp>

namespace FlexFlow {

SearchResult
    mcmc_over_mapped_pcg(ParallelComputationGraph &pcg,
                         RuntimeOnlyCostEstimator const &cost_estimator,
                         MachineSpecification const &resources,
                         MCMCOverMappedPCGConfig const &search_config) {

  std::vector<Substitution> substitutions = get_substitution_set(resources);
  MachineMapping random_mapping = assert_unwrap(
      get_random_mapping(pcg, resources, search_config.device_type));
  SearchResult starting_state = SearchResult{pcg, random_mapping};

  auto sampler = [&](SearchResult mapped_pcg) -> std::optional<SearchResult> {
    // applies substitution with substitution_frequency probability
    // applies machine mapping mutation with (1 - substitution_frequency)
    // probability
    ASSERT(search_config.substitution_frequency >= 0 &&
           search_config.substitution_frequency <= 1);
    if (randf() < search_config.substitution_frequency) {
      Substitution random_substitution =
          assert_unwrap(get_random_substitution(resources));
      std::optional<PCGPatternMatch> maybe_pattern_match =
          get_random_pattern_match(random_substitution.pcg_pattern,
                                   sub_pcg_from_full_pcg(mapped_pcg.pcg));
      return transform(maybe_pattern_match, [&](PCGPatternMatch match) {
        return apply_substitution_and_update_machine_mapping(
            mapped_pcg, random_substitution, match);
      });
    } else {
      MachineMapping new_machine_mapping = assert_unwrap(get_random_mutation(
          mapped_pcg, resources, search_config.device_type));
      return SearchResult{mapped_pcg.pcg, new_machine_mapping};
    }
  };

  auto cost = [&](SearchResult mapped_pcg) -> float {
    return task_simulator_estimate_forward_pass_time(mapped_pcg.pcg,
                                                     cost_estimator,
                                                     mapped_pcg.machine_mapping,
                                                     resources)
        .unwrap_milliseconds();
  };

  GenericMCMCConfig config =
      GenericMCMCConfig{/*temperature*/ search_config.temperature,
                        /*num_iterations*/ search_config.num_iterations};

  SearchResult result = run_mcmc(starting_state, sampler, cost, config);

  return result;
}

} // namespace FlexFlow
