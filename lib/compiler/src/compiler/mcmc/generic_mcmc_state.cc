#include "compiler/mcmc/generic_mcmc_state.h"
#include "utils/archetypes/ordered_value_type.h"
#include "utils/archetypes/value_type.h"

namespace FlexFlow {
using State = value_type<0>;
using Score = ordered_value_type<1>;

template struct Generic_MCMC_state<State, Score>;
template struct Generic_MCMC_state<State, float>;

} // namespace FlexFlow
