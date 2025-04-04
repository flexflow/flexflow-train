#ifndef _FLEXFLOW_COMPILER_MCMC_GENERIC_MCMC_ALGORITHM_H
#define _FLEXFLOW_COMPILER_MCMC_GENERIC_MCMC_ALGORITHM_H

#include "compiler/mcmc/generic_mcmc_config.dtg.h"
#include "compiler/mcmc/generic_mcmc_state.h"
#include "utils/nonnegative_int/nonnegative_range.h"
#include "utils/random_utils.h"
#include <optional>

namespace FlexFlow {

bool mcmc_accept(float delta, float temperature) {
  return delta < 0 || (randf() < exp(-delta / temperature));
}

template <typename State, typename ScoringFunc>
void modify_state(Generic_MCMC_state<State, float> &best_state,
                  Generic_MCMC_state<State, float> &current_state,
                  State candidate,
                  ScoringFunc scorer,
                  float temperature) {
  float best_estimate = best_state.get_score();
  float new_estimate = scorer(candidate);
  float runtime_delta = new_estimate - best_estimate;
  if (mcmc_accept(runtime_delta, temperature)) {
    current_state = Generic_MCMC_state<State, float>(candidate, new_estimate);
    if (runtime_delta < 0) {
      best_state = current_state;
    }
  }
}

// GeneratingFunc : State -> nn_int -> std::optional<State>
// ScoringFunc : State -> float

template <typename State, typename GeneratingFunc, typename ScoringFunc>
Generic_MCMC_state<State, float>
    minimize_score(State const &starting_state,
                   GeneratingFunc const &generator,
                   ScoringFunc const &scorer,
                   GenericMCMCConfig const &search_config) {
  using MCMCState = Generic_MCMC_state<State, float>;
  MCMCState best_state = MCMCState(starting_state, scorer(starting_state));
  MCMCState current_state = best_state;
  for (nonnegative_int i : nonnegative_range(search_config.num_iterations)) {
    std::optional<State> candidate = generator(current_state.get_state(), i);
    if (candidate != std::nullopt) {
      modify_state(best_state,
                   current_state,
                   candidate.value(),
                   scorer,
                   search_config.temperature);
    }
  }
  return best_state;
}

} // namespace FlexFlow

#endif
