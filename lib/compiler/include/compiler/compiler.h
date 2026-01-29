#ifndef _FLEXFLOW_COMPILER_COMPILER_H
#define _FLEXFLOW_COMPILER_COMPILER_H

#include "compiler/algorithm_config.dtg.h"
#include "compiler/cost_estimator/cost_estimator.h"
#include "compiler/search_result.dtg.h"
#include "pcg/machine_specification.dtg.h"
#include "pcg/computation_graph.h"

namespace FlexFlow {

SearchResult optimize(ComputationGraph const &,
                      MachineSpecification const &,
                      CostEstimator const &,
                      AlgorithmConfig const &);

} // namespace FlexFlow

#endif
