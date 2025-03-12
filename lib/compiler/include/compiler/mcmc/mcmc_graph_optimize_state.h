#ifndef _FLEXFLOW_COMPILER_MCMC_ALGORITHM_STATE_H
#define _FLEXFLOW_COMPILER_MCMC_ALGORITHM_STATE_H

#include "compiler/search_result.h"
#include "pcg/parallel_computation_graph/parallel_computation_graph.h"

namespace FlexFlow {

struct MCMCOptimizeState {
  MCMCOptimizeState() = delete;
  explicit MCMCOptimizeState(SearchResult const &mapped_pcg, float runtime);

  SearchResult mapped_pcg;
  float runtime;

  bool operator==(MCMCOptimizeState const &other) const;
  bool operator!=(MCMCOptimizeState const &other) const;
  bool operator<(MCMCOptimizeState const &other) const;
};

std::string format_as(MCMCOptimizeState const &);
std::ostream &operator<<(std::ostream &, MCMCOptimizeState const &);

} // namespace FlexFlow

namespace std {

template <>
struct hash<::FlexFlow::MCMCOptimizeState> {
  size_t operator()(::FlexFlow::MCMCOptimizeState const &) const;
};

} // namespace std

#endif
