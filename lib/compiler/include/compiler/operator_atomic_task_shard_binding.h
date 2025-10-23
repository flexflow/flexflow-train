#ifndef _FLEXFLOW_LIB_COMPILER_INCLUDE_COMPILER_OPERATOR_ATOMIC_TASK_SHARD_BINDING_H
#define _FLEXFLOW_LIB_COMPILER_INCLUDE_COMPILER_OPERATOR_ATOMIC_TASK_SHARD_BINDING_H

#include "compiler/operator_atomic_task_shard_binding.dtg.h"
#include "compiler/task_signature_tensor_key.dtg.h"
#include "op-attrs/computation_graph_op_attrs.dtg.h"
#include "op-attrs/parallel_tensor_space_coordinate.dtg.h"
#include "pcg/machine_view.dtg.h"

namespace FlexFlow {

OperatorAtomicTaskShardBinding
  operator_atomic_task_shard_binding_from_machine_view(ComputationGraphOpAttrs const &,
                                                       std::vector<ParallelTensorDimDegrees> const &,
                                                       MachineView const &,
                                                       MachineSpaceCoordinate const &);

} // namespace FlexFlow

#endif
