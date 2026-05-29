#ifndef _FLEXFLOW_LIB_PCG_INCLUDE_PCG_MAPPED_PARALLEL_COMPUTATION_GRAPH_MAPPED_PARALLEL_LAYER_INVOCATION_INFO_H
#define _FLEXFLOW_LIB_PCG_INCLUDE_PCG_MAPPED_PARALLEL_COMPUTATION_GRAPH_MAPPED_PARALLEL_LAYER_INVOCATION_INFO_H

#include "pcg/mapped_parallel_computation_graph/mapped_parallel_layer_invocation_info.dtg.h"
#include "pcg/parallel_computation_graph/parallel_layer_invocation_info.dtg.h"
#include "pcg/mapped_parallel_computation_graph/mapped_operator_task_group.h"

namespace FlexFlow {

MappedParallelLayerInvocationInfo
  mapped_parallel_layer_invocation_info_from_pcg_invocation_and_mapping(
    ParallelLayerInvocationInfo const &,
    MappedOperatorTaskGroup const &);

} // namespace FlexFlow

#endif
