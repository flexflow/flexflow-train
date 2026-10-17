#include "op-attrs/ops/input.h"
#include "op-attrs/operator_space_to_parallel_tensor_space_mapping.h"
#include "op-attrs/operator_task_space.h"
#include "op-attrs/parallel_tensor_dim_degrees.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "op-attrs/parallel_tensor_space_coordinate.h"
#include "op-attrs/task_space_coordinate.h"
#include "op-attrs/tensor_dims.h"

namespace FlexFlow {

TensorShape input_get_output_shape(InputAttrs const &attrs) {
  return attrs.tensor_shape;
}

ParallelTensorDimDegrees
    input_get_output_parallel_dim_degrees(InputAttrs const &attrs) {
  return trivial_degrees_for_tensor_dims(attrs.tensor_shape.dims);
}

ParallelTensorShape input_get_output_parallel_shape(InputAttrs const &attrs) {
  return lift_shape_to_parallel(attrs.tensor_shape);
}

StandardOperatorTaskGroup input_get_task_group(InputAttrs const &attrs) {
  return StandardOperatorTaskGroup{
      std::set{
          AbstractedOperatorAtomicTaskShardBinding{
              /*tensor_coords=*/std::map<TensorSlotName,
                                         ParallelTensorSpaceCoordinate>{
                  {
                      TensorSlotName::OUTPUT,
                      trivial_parallel_tensor_space_coordinate_for_num_tensor_dims(
                          get_num_dims(attrs.tensor_shape.dims)),
                  },
              },
              /*task_coord=*/trivial_task_space_coordinate(),
          },
      },
  };
}

OperatorTaskSpace input_get_operator_task_space(InputAttrs const &) {
  return trivial_op_task_space();
}

OperatorSpaceToParallelTensorSpaceMapping
    input_get_operator_to_output_mapping(InputAttrs const &attrs) {

  return empty_operator_space_to_ptensor_space_map();
}

} // namespace FlexFlow
