#ifndef _FLEXFLOW_COMPILER_MCMC_ALGORITHM_H
#define _FLEXFLOW_COMPILER_MCMC_ALGORITHM_H

#include "compiler/cost_estimator/cost_estimator.h"
#include "compiler/mcmc/mcmc_search_config.dtg.h"
#include "compiler/search_result.dtg.h"
#include "pcg/computation_graph.h"
#include "pcg/machine_specification.dtg.h"
#include "pcg/parallel_computation_graph/parallel_computation_graph.dtg.h"
#include "substitutions/sub_parallel_computation_graph.h"
#include "substitutions/substitution.h"

namespace FlexFlow {

SearchResult mcmc_graph_optimize(ParallelComputationGraph &pcg,
                                 CostEstimator const &cost_estimator,
                                 MachineSpecification const &resources,
                                 MCMCSearchConfig const &search_config);

} // namespace FlexFlow

#endif
