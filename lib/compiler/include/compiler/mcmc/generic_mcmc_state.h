
#ifndef _FLEXFLOW_COMPILER_MCMC_GENERIC_MCMC_STATE_H
#define _FLEXFLOW_COMPILER_MCMC_GENERIC_MCMC_STATE_H
#include "utils/nonnegative_int/nonnegative_int.h"

namespace FlexFlow {

template <typename State, typename Score>
struct Generic_MCMC_state {
public:
  Generic_MCMC_state(State const &state, Score const &score)
      : state(state), score(score) {}

  State const &get_state() const {
    return state;
  }
  Score const &get_score() const {
    return score;
  }

private:
  State state;
  Score score;
};

} // namespace FlexFlow

#endif
