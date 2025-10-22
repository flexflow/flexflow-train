#ifndef _FLEXFLOW_COMPILER_MCMC_OVER_MAPPED_PCG_H
#define _FLEXFLOW_COMPILER_MCMC_OVER_MAPPED_PCG_H

#include "compiler/cost_estimator/runtime_only_cost_estimator.h"
#include "compiler/mcmc/mcmc_over_mapped_pcg_config.dtg.h"
#include "compiler/search_result.dtg.h"
#include "pcg/computation_graph.h"
#include "pcg/machine_specification.dtg.h"
#include "pcg/parallel_computation_graph/parallel_computation_graph.dtg.h"
#include "substitutions/sub_parallel_computation_graph.h"
#include "substitutions/substitution.h"

namespace FlexFlow {

SearchResult mcmc_graph_optimize(ParallelComputationGraph &pcg,
                                 RuntimeOnlyCostEstimator const &cost_estimator,
                                 MachineSpecification const &resources,
                                 MCMCOverMappedPCGConfig const &search_config);

} // namespace FlexFlow

#endif
