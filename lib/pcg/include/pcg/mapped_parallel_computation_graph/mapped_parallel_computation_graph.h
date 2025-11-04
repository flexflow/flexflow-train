#ifndef _FLEXFLOW_LIB_PCG_INCLUDE_PCG_MAPPED_PARALLEL_COMPUTATION_GRAPH_MAPPED_PARALLEL_COMPUTATION_GRAPH_H
#define _FLEXFLOW_LIB_PCG_INCLUDE_PCG_MAPPED_PARALLEL_COMPUTATION_GRAPH_MAPPED_PARALLEL_COMPUTATION_GRAPH_H

#include "pcg/mapped_parallel_computation_graph/mapped_parallel_computation_graph.dtg.h"

namespace FlexFlow {

std::string format_as(MappedParallelComputationGraph const &);
std::ostream &operator<<(std::ostream &, MappedParallelComputationGraph const &);

} // namespace FlexFlow

#endif
