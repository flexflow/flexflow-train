#ifndef _FLEXFLOW_LIB_PCG_INCLUDE_PCG_FILE_FORMAT_V1_V1_MAPPED_PARALLEL_COMPUTATION_GRAPH_H
#define _FLEXFLOW_LIB_PCG_INCLUDE_PCG_FILE_FORMAT_V1_V1_MAPPED_PARALLEL_COMPUTATION_GRAPH_H

#include "pcg/file_format/v1/v1_mapped_parallel_computation_graph.dtg.h"
#include "pcg/mapped_parallel_computation_graph/mapped_parallel_computation_graph.dtg.h"

namespace FlexFlow {

V1MappedParallelComputationGraph to_v1(MappedParallelComputationGraph const &);
MappedParallelComputationGraph
    from_v1(V1MappedParallelComputationGraph const &);

} // namespace FlexFlow

#endif
