#ifndef _FLEXFLOW_LIB_COMPILER_INCLUDE_COMPILER_UNITY_ALGORITHM_GRAPH_OPTIMIZE_STATE_H
#define _FLEXFLOW_LIB_COMPILER_INCLUDE_COMPILER_UNITY_ALGORITHM_GRAPH_OPTIMIZE_STATE_H

#include "compiler/graph_optimize_result.dtg.h"
#include "compiler/machine_mapping/machine_mapping_result.dtg.h"
#include "compiler/series_parallel/pcg/pcg_binary_sp_decomposition.dtg.h"
#include "utils/units/milliseconds_t.h"

namespace FlexFlow {

struct GraphOptimizeState {
  GraphOptimizeState() = delete;
  explicit GraphOptimizeState(ParallelComputationGraph const &parallel_computation_graph,
                              milliseconds_t runtime);

  bool operator==(GraphOptimizeState const &other) const;
  bool operator!=(GraphOptimizeState const &other) const;
  bool operator<(GraphOptimizeState const &other) const;

public:
  ParallelComputationGraph pcg;
  milliseconds_t runtime;
};

std::string format_as(GraphOptimizeState const &);
std::ostream &operator<<(std::ostream &, GraphOptimizeState const &);

// TODO(@lockshaw)(#pr): Delete this if still unused
// std::optional<GraphOptimizeState>
//   graph_optimize_state_from_machine_mapping_result(ParallelComputationGraph const &, 
//                                                    PCGBinarySPDecomposition const &,
//                                                    MachineMappingResult const &);

} // namespace FlexFlow

namespace std {

template <>
struct hash<::FlexFlow::GraphOptimizeState> {
  size_t operator()(::FlexFlow::GraphOptimizeState const &) const;
};

} // namespace std

#endif
