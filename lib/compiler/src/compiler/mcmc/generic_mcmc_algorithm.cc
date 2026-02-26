#include "compiler/mcmc/generic_mcmc_algorithm.h"
#include "utils/archetypes/value_type.h"

namespace FlexFlow {

using State = value_type<0>;
using SamplingFn = std::function<std::optional<State>(State)>;
using CostFn = std::function<float(State)>;

template State run_mcmc(State const &starting_state,
                        SamplingFn const &sampler,
                        CostFn const &cost,
                        GenericMCMCConfig const &search_config);

} // namespace FlexFlow
