#include "op-attrs/ops/concat.h"
#include "op-attrs/ff_ordered/ff_ordered_enumerate.h"
#include "op-attrs/ff_ordered/ff_ordered_from_map.h"
#include "op-attrs/parallel_tensor_dim_degrees.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "op-attrs/standard_operator_task_group.h"
#include "op-attrs/task_space_coordinate.h"
#include "op-attrs/tensor_dims.h"
#include "op-attrs/tensor_shape.h"
#include "op-attrs/tensor_slot_name.h"
#include "utils/containers/all_of.h"
#include "utils/containers/are_all_same.h"
#include "utils/containers/generate_map.h"
#include "utils/containers/require_all_same1.h"
#include "utils/containers/slice.h"
#include "utils/containers/sum.h"
#include "utils/containers/transform.h"
#include "utils/fmt/map.h"

namespace FlexFlow {

std::vector<TensorSlotName>
    concat_get_input_slot_names(ConcatAttrs const &attrs) {
  return slice(get_variadic_inputs_slot_name_sequence(),
               0,
               attrs.num_inputs.int_from_int_ge_two());
}

TensorShape concat_get_output_shape(ConcatAttrs const &attrs,
                                    std::vector<TensorShape> const &inputs) {
  ASSERT(attrs.num_inputs == inputs.size());

  auto get_non_axis_dims =
      [&](TensorShape const &s) -> std::map<ff_dim_t, positive_int> {
    std::map<ff_dim_t, positive_int> dim_sizes =
        ff_ordered_enumerate(ff_ordered(s.dims));
    dim_sizes.erase(attrs.axis);
    return dim_sizes;
  };

  num_tensor_dims_t input_num_dims = require_all_same1(transform(
      inputs, [](TensorShape const &s) { return get_num_dims(s.dims); }));

  ASSERT(attrs.axis.value < input_num_dims.int_from_num_tensor_dims());

  std::map<ff_dim_t, positive_int> non_axis_dims =
      require_all_same1(transform(inputs, get_non_axis_dims));

  std::vector<positive_int> axis_dim_sizes =
      transform(inputs, [&](TensorShape const &s) {
        return dim_at_idx(s.dims, relative_ff_dim_t_from_ff_dim_t(attrs.axis));
      });

  positive_int output_axis_dim_size = sum(axis_dim_sizes);

  non_axis_dims.insert({attrs.axis, output_axis_dim_size});

  DataType datatype = require_all_same1(
      transform(inputs, [](TensorShape const &s) { return s.data_type; }));

  return TensorShape{
      TensorDims{
          ff_ordered_from_map(non_axis_dims),
      },
      datatype,
  };
}

ParallelTensorDimDegrees concat_get_output_parallel_dim_degrees(
    ConcatAttrs const &attrs,
    std::vector<ParallelTensorDimDegrees> const &inputs) {
  SumDegree sum_degree = SumDegree{
      require_all_same1(transform(inputs,
                                  [&](ParallelTensorDimDegrees const &d)
                                      -> SumDegree { return d.sum_degree; })),
  };

  DiscardCopyDegree discard_copy_degree = DiscardCopyDegree{
      require_all_same1(transform(
          inputs,
          [&](ParallelTensorDimDegrees const &d) -> DiscardCopyDegree {
            return d.discard_copy_degree;
          })),
  };

  ASSERT(all_of(inputs,
                [&](ParallelTensorDimDegrees const &d) -> bool {
                  return d.shard_degrees.at(attrs.axis) == 1;
                }),
         "get_output_shape for Concat expected input tensors to have parallel "
         "degree 1 in the concat axis dimension",
         inputs);

  return require_all_same1(inputs);
}

ParallelTensorShape concat_get_output_parallel_shape(
    ConcatAttrs const &attrs,
    std::vector<ParallelTensorShape> const &input_shapes) {

  TensorShape output_shape = concat_get_output_shape(
      attrs, transform(input_shapes, get_reduced_shape));

  ParallelTensorDimDegrees output_degrees =
      concat_get_output_parallel_dim_degrees(
          attrs, transform(input_shapes, get_parallel_degrees));

  return lift_shape_to_parallel_with_degrees(output_shape, output_degrees);
}

StandardOperatorTaskGroup concat_get_task_group(
    ConcatAttrs const &attrs,
    std::vector<ParallelTensorDimDegrees> const &inputs_degrees) {
  ParallelTensorDimDegrees output_degrees =
      concat_get_output_parallel_dim_degrees(attrs, inputs_degrees);

  std::vector<TensorSlotName> input_slot_names =
      concat_get_input_slot_names(attrs);

  ParallelTensorDimDegrees input_degrees = require_all_same1(inputs_degrees);

  return StandardOperatorTaskGroup{
      transform(
          get_parallel_tensor_space_coordinates(input_degrees),
          [&](ParallelTensorSpaceCoordinate const &input_coord)
              -> AbstractedOperatorAtomicTaskShardBinding {
            ParallelTensorSpaceCoordinate output_coord = input_coord;

            return AbstractedOperatorAtomicTaskShardBinding{
                /*tensor_coords=*/binary_merge_disjoint_maps(
                    generate_map(
                        input_slot_names,
                        [&](TensorSlotName) -> ParallelTensorSpaceCoordinate {
                          return input_coord;
                        }),
                    std::map<TensorSlotName, ParallelTensorSpaceCoordinate>{
                        {
                            TensorSlotName::OUTPUT,
                            output_coord,
                        },
                    }),
                /*task_coord=*/
                task_coord_matching_parallel_tensor_space_coordinate(
                    output_coord, output_degrees),
            };
          }),
  };
}

} // namespace FlexFlow
