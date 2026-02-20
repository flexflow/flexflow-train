#ifndef _FLEXFLOW_LIB_PCG_INCLUDE_PCG_MAPPED_PARALLEL_COMPUTATION_GRAPH_OPERATOR_ATOMIC_TASK_SHARD_BINDING_H
#define _FLEXFLOW_LIB_PCG_INCLUDE_PCG_MAPPED_PARALLEL_COMPUTATION_GRAPH_OPERATOR_ATOMIC_TASK_SHARD_BINDING_H

#include "op-attrs/tensor_role.dtg.h"
#include "op-attrs/tensor_slot_name.dtg.h"
#include "pcg/mapped_parallel_computation_graph/operator_atomic_task_shard_binding.dtg.h"

namespace FlexFlow {

ParallelTensorSpaceCoordinate
    ptensor_space_coord_for_slot_name(OperatorAtomicTaskShardBinding const &,
                                      TensorSlotName const &);

} // namespace FlexFlow

#endif
