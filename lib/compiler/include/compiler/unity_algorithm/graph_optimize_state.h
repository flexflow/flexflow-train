#ifndef _FLEXFLOW_COMPILER_MCMC_STATE_H
#define _FLEXFLOW_COMPILER_MCMC_STATE_H

#include "pcg/parallel_computation_graph/parallel_computation_graph.h"

namespace FlexFlow {

struct GraphOptimizeState {
  GraphOptimizeState() = delete;
  explicit GraphOptimizeState(ParallelComputationGraph const &pcg,
                              float runtime);

  ParallelComputationGraph pcg;
  float runtime_with_optimal_mm;

  bool operator==(GraphOptimizeState const &other) const;
  bool operator!=(GraphOptimizeState const &other) const;
  bool operator<(GraphOptimizeState const &other) const;
};

} // namespace FlexFlow

namespace std {

template <>
struct hash<::FlexFlow::GraphOptimizeState> {
  size_t operator()(::FlexFlow::GraphOptimizeState const &) const;
};

} // namespace std

#endif
