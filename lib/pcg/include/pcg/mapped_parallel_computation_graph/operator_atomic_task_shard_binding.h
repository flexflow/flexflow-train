#ifndef _FLEXFLOW_LIB_PCG_INCLUDE_PCG_MAPPED_PARALLEL_COMPUTATION_GRAPH_OPERATOR_ATOMIC_TASK_SHARD_BINDING_H
#define _FLEXFLOW_LIB_PCG_INCLUDE_PCG_MAPPED_PARALLEL_COMPUTATION_GRAPH_OPERATOR_ATOMIC_TASK_SHARD_BINDING_H

#include "op-attrs/tensor_role.dtg.h"
#include "pcg/mapped_parallel_computation_graph/operator_atomic_task_shard_binding.dtg.h"
#include "pcg/mapped_parallel_computation_graph/task_signature_tensor_key.dtg.h"

namespace FlexFlow {

std::vector<ParallelTensorSpaceCoordinate>
  ptensor_space_coords_for_role(OperatorAtomicTaskShardBinding const &,
                                TensorRole);

ParallelTensorSpaceCoordinate
  ptensor_space_coord_for_key(OperatorAtomicTaskShardBinding const &,
                              TaskSignatureTensorKey const &);


} // namespace FlexFlow

#endif
