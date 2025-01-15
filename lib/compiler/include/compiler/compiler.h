#ifndef _FLEXFLOW_COMPILER_COMPILER_H
#define _FLEXFLOW_COMPILER_COMPILER_H

#include "compiler/algorithm_config.dtg.h"
#include "compiler/cost_estimator/cost_estimator.h"
#include "compiler/search_result.dtg.h"
#include "pcg/machine_specification.dtg.h"

namespace FlexFlow {

enum class SearchAlgorithm {
  DATA_PARALLEL,
  UNITY,
};

SearchResult optimize(ComputationGraph const &,
                      MachineSpecification const &,
                      CostEstimator const &,
                      SearchAlgorithm,
                      AlgorithmConfig const &,
                      DeviceType);

} // namespace FlexFlow

#endif
