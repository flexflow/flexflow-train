#ifndef _FLEXFLOW_LIB_COMPILER_INCLUDE_COMPILER_OPERATOR_TASK_SIGNATURE_H
#define _FLEXFLOW_LIB_COMPILER_INCLUDE_COMPILER_OPERATOR_TASK_SIGNATURE_H

#include "compiler/operator_task_signature.dtg.h"
#include "compiler/task_signature_tensor_key.dtg.h"
#include "op-attrs/computation_graph_op_attrs.dtg.h"
#include "op-attrs/parallel_tensor_space_coordinate.dtg.h"
#include "pcg/machine_view.dtg.h"

namespace FlexFlow {

OperatorTaskSignature
  operator_task_signature_from_machine_view(ComputationGraphOpAttrs const &,
                                            std::vector<ParallelTensorDimDegrees> const &,
                                            MachineView const &,
                                            MachineSpaceCoordinate const &);

std::vector<ParallelTensorSpaceCoordinate>
  ptensor_space_coords_for_role(OperatorTaskSignature const &,
                                TensorRole);

ParallelTensorSpaceCoordinate
  ptensor_space_coord_for_key(OperatorTaskSignature const &,
                              TaskSignatureTensorKey const &);

} // namespace FlexFlow

#endif
