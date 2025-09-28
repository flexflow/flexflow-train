#include "compiler/operator_task_signature.h"
#include "op-attrs/get_operator_space_to_parallel_tensor_space_mappings.h"
#include "op-attrs/get_operator_task_space.h"
#include "op-attrs/operator_space_to_parallel_tensor_space_mapping.h"
#include "pcg/machine_view.h"
#include "utils/containers/at_idx.h"
#include <libassert/assert.hpp>

namespace FlexFlow {

OperatorTaskSignature
  operator_task_signature_from_machine_view(ComputationGraphOpAttrs const &op_attrs,
                                            std::vector<ParallelTensorDimDegrees> const &inputs_dim_degrees,
                                            MachineView const &machine_view,
                                            MachineSpaceCoordinate const &machine_space_coord) {
  OperatorTaskSpace op_task_space = get_operator_task_space(op_attrs, inputs_dim_degrees);

  TaskSpaceCoordinate task_space_coord = mv_task_space_coord_for_machine_space_coord(
    machine_view,
    op_task_space,
    machine_space_coord);

  auto get_ptensor_coords = [&](TensorRole const &tensor_role) {
    std::vector<OperatorSpaceToParallelTensorSpaceMapping>
        mappings = get_operator_to_ptensor_mappings_for_role(op_attrs, inputs_dim_degrees, tensor_role);

    std::vector<ParallelTensorSpaceCoordinate>
      ptensor_coords = transform(mappings,
                               [&](OperatorSpaceToParallelTensorSpaceMapping const &mapping) {
                                 return ptensor_coord_for_task_space_coord(mapping, task_space_coord);
                               });

    return ptensor_coords;
  };

  return OperatorTaskSignature{
    /*inputs=*/get_ptensor_coords(TensorRole::INPUT),
    /*weights=*/get_ptensor_coords(TensorRole::WEIGHT),
    /*outputs=*/get_ptensor_coords(TensorRole::OUTPUT),
  };
}

std::vector<ParallelTensorSpaceCoordinate>
  ptensor_space_coords_for_role(OperatorTaskSignature const &op_task_signature,
                                TensorRole tensor_role) {
  switch (tensor_role) {
    case TensorRole::INPUT:
      return op_task_signature.inputs;
    case TensorRole::WEIGHT:
      return op_task_signature.weights;
    case TensorRole::OUTPUT:
      return op_task_signature.outputs;
    default: 
      PANIC("Unhandled TensorRole", tensor_role); 
  };
}

ParallelTensorSpaceCoordinate
  ptensor_space_coord_for_key(OperatorTaskSignature const &op_task_signature,
                              TaskSignatureTensorKey const &tensor_key) {
  return at_idx(
    ptensor_space_coords_for_role(op_task_signature, tensor_key.tensor_role),
    tensor_key.idx);
}

} // namespace FlexFlow
