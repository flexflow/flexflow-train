#include "op-attrs/ops/flat.h"
#include "op-attrs/ff_dim_t.h"
#include "op-attrs/ff_ordered/ff_ordered_concat.h"
#include "op-attrs/ff_ordered/ff_ordered_slice.h"
#include "op-attrs/ff_ordered/ff_ordered_slice_inclusive.h"
#include "op-attrs/num_ptensor_shard_dims_t.h"
#include "op-attrs/num_tensor_dims_t.h"
#include "op-attrs/operator_task_space.dtg.h"
#include "op-attrs/operator_task_space.h"
#include "op-attrs/parallel_tensor_dim_degrees.h"
#include "op-attrs/parallel_tensor_dim_idx_t.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "op-attrs/parallel_tensor_space_coordinate.h"
#include "op-attrs/standard_operator_task_group.h"
#include "op-attrs/task_space_coordinate.h"
#include "op-attrs/tensor_dims.h"
#include "utils/containers/all_of.h"
#include "utils/containers/merge_disjoint_sets.h"
#include "utils/containers/product.h"
#include "utils/containers/set_of.h"
#include "utils/optional.h"
#include "utils/orthotope/eq_projection.h"
#include "utils/orthotope/orthotope_bounded_coord.h"
#include <cassert>

namespace FlexFlow {

TensorShape flat_get_output_shape(FlatAttrs const &attrs,
                                  TensorShape const &input_shape) {
  FFOrdered<positive_int> leading_dims = ff_ordered_slice(
      ff_ordered(input_shape.dims), ff_dim_t{0_n}, attrs.start_dim);
  FFOrdered<positive_int> flattened_dims = ff_ordered_slice_inclusive(
      ff_ordered(input_shape.dims), attrs.start_dim, attrs.end_dim);
  FFOrdered<positive_int> trailing_dims =
      ff_ordered_slice(ff_ordered(input_shape.dims),
                       add_to_ff_dim(attrs.end_dim, 1),
                       std::nullopt);

  if (flattened_dims.empty()) {
    return input_shape;
  }

  return TensorShape{
      TensorDims{
          ff_ordered_concat(std::vector{
              leading_dims,
              FFOrdered{product(flattened_dims)},
              trailing_dims,
          }),
      },
      input_shape.data_type,
  };
}

ParallelTensorDimDegrees flat_get_output_parallel_dim_degrees(
    FlatAttrs const &attrs, ParallelTensorDimDegrees const &input_degrees) {
  FFOrdered<positive_int> flattened_dim_degrees = ff_ordered_slice_inclusive(
      input_degrees.shard_degrees, attrs.start_dim, attrs.end_dim);

  if (flattened_dim_degrees.empty()) {
    return input_degrees;
  }

  ASSERT(all_of(flattened_dim_degrees,
                [](positive_int degree) { return degree == 1; }),
         "flat_get_output_parallel_dim_degrees expected all shard degrees of "
         "flattened dimensions to be 1",
         attrs,
         input_degrees,
         flattened_dim_degrees);

  return ParallelTensorDimDegrees{
      /*sum_degree=*/input_degrees.sum_degree,
      /*discard_copy_degree=*/input_degrees.discard_copy_degree,
      /*shard_degrees=*/
      ff_ordered_concat(std::vector{
          ff_ordered_slice(
              input_degrees.shard_degrees, ff_dim_t{0_n}, attrs.start_dim),
          FFOrdered{product(flattened_dim_degrees)},
          ff_ordered_slice(input_degrees.shard_degrees,
                           add_to_ff_dim(attrs.end_dim, 1),
                           std::nullopt),
      }),
  };
}

ParallelTensorShape
    flat_get_output_parallel_shape(FlatAttrs const &attrs,
                                   ParallelTensorShape const &input_shape) {
  TensorShape unpar =
      flat_get_output_shape(attrs, get_reduced_shape(input_shape));

  ParallelTensorDimDegrees degrees = flat_get_output_parallel_dim_degrees(
      attrs, get_parallel_degrees(input_shape));

  return lift_shape_to_parallel_with_degrees(unpar, degrees);
}

StandardOperatorTaskGroup
    flat_get_task_group(FlatAttrs const &attrs,
                        ParallelTensorDimDegrees const &input_degrees) {
  ParallelTensorDimDegrees output_degrees =
      flat_get_output_parallel_dim_degrees(attrs, input_degrees);

  auto to_shard_dims =
      [&](std::set<ff_dim_t> const &ds) -> std::set<parallel_tensor_dim_idx_t> {
    return transform(ds, [](ff_dim_t d) -> parallel_tensor_dim_idx_t {
      return shard_dim_idx(d);
    });
  };

  return StandardOperatorTaskGroup{
      transform(
          get_parallel_tensor_space_coordinates(input_degrees),
          [&](ParallelTensorSpaceCoordinate const &input_coord)
              -> AbstractedOperatorAtomicTaskShardBinding {
            std::set<parallel_tensor_dim_idx_t> leading_dims =
                to_shard_dims(set_of(
                    ff_dim_range2_exclusive(ff_dim_t{0_n}, attrs.start_dim)));

            std::set<parallel_tensor_dim_idx_t> flattened_dims =
                to_shard_dims(set_of(
                    ff_dim_range2_inclusive(attrs.start_dim, attrs.end_dim)));

            std::set<parallel_tensor_dim_idx_t> trailing_dims =
                to_shard_dims(set_of(ff_dim_range2_exclusive(
                    add_to_ff_dim(attrs.end_dim, 1),
                    ff_dim_t{num_elements(input_degrees.shard_degrees)})));

            std::set<parallel_tensor_dim_idx_t> reduction_dims = {
                sum_dim_idx(),
                discard_copy_dim_idx(),
            };

            ASSERT(merge_disjoint_sets(std::vector{reduction_dims,
                                                   leading_dims,
                                                   flattened_dims,
                                                   trailing_dims}) ==
                   get_parallel_tensor_dim_indices(input_degrees));

            BoundedComponent sum_component = bounded_component_for_ptensor_dim(
                input_degrees, input_coord, sum_dim_idx());

            BoundedComponent discard_copy_component =
                bounded_component_for_ptensor_dim(
                    input_degrees, input_coord, sum_dim_idx());

            OrthotopeBoundedCoord leading =
                orthotope_bounded_coord_for_ptensor_dims(
                    input_degrees, input_coord, leading_dims);

            OrthotopeBoundedCoord flattened =
                orthotope_bounded_coord_for_ptensor_dims(
                    input_degrees, input_coord, flattened_dims);

            OrthotopeBoundedCoord trailing =
                orthotope_bounded_coord_for_ptensor_dims(
                    input_degrees, input_coord, trailing_dims);

            ParallelTensorSpaceCoordinate output_coord =
                parallel_tensor_space_coordinate_from_bounded_orthotope_components(
                    /*sum_component=*/sum_component,
                    /*discard_copy_component=*/discard_copy_component,
                    /*shard_components=*/
                    orthotope_bounded_coord_product(
                        leading,
                        lift_bounded_component(assert_unwrap(
                            flatten_orthotope_bounded_coord(flattened))),
                        trailing));

            ASSERT(parallel_tensor_space_contains_coord(output_degrees,
                                                        output_coord));

            return AbstractedOperatorAtomicTaskShardBinding{
                /*tensor_corods=*/{
                    {
                        TensorSlotName::INPUT,
                        input_coord,
                    },
                    {TensorSlotName::OUTPUT, output_coord},
                },
                /*task_coord=*/
                task_coord_matching_parallel_tensor_space_coordinate(
                    output_coord, output_degrees),
            };
          }),
  };
}

} // namespace FlexFlow
