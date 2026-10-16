#include "op-attrs/ops/batch_matmul.h"
#include "op-attrs/ff_ordered/ff_ordered_concat.h"
#include "op-attrs/ff_ordered/ff_ordered_slice.h"
#include "op-attrs/parallel_tensor_dim_degrees.h"
#include "op-attrs/parallel_tensor_dim_idx_t.h"
#include "op-attrs/parallel_tensor_dims.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "op-attrs/parallel_tensor_space_coordinate.h"
#include "op-attrs/task_space_coordinate.h"
#include "op-attrs/tensor_dims.h"
#include "utils/containers/binary_cartesian_product.h"
#include "utils/containers/require_same.h"
#include "utils/containers/set_filtrans.h"
#include "utils/optional.h"
#include "utils/orthotope/bounded_component.h"
#include "utils/orthotope/orthotope_bounded_coord.h"
#include <libassert/assert.hpp>

namespace FlexFlow {

TensorShape batch_matmul_get_output_shape(BatchMatmulAttrs const &,
                                          TensorShape const &lhs,
                                          TensorShape const &rhs) {
  positive_int lhs_row_dim = dim_at_idx(lhs.dims, relative_ff_dim_t{-2});
  positive_int lhs_col_dim = dim_at_idx(lhs.dims, relative_ff_dim_t{-1});

  positive_int rhs_row_dim = dim_at_idx(rhs.dims, relative_ff_dim_t{-2});
  positive_int rhs_col_dim = dim_at_idx(rhs.dims, relative_ff_dim_t{-1});

  ASSERT(lhs_col_dim == rhs_row_dim);

  auto get_leading_dims = [](TensorShape const &s) -> FFOrdered<positive_int> {
    return ff_ordered_slice(
        s.dims.ff_ordered, relative_ff_dim_t{0}, relative_ff_dim_t{-2});
  };

  FFOrdered<positive_int> leading_dims =
      require_same(get_leading_dims(lhs), get_leading_dims(rhs));

  return TensorShape{
      TensorDims{
          ff_ordered_concat(leading_dims,
                            FFOrdered<positive_int>{
                                lhs_row_dim,
                                rhs_col_dim,
                            }),
      },
      /*data_type=*/require_same(lhs.data_type, rhs.data_type),
  };
}

ParallelTensorDimDegrees batch_matmul_get_output_parallel_dim_degrees(
    BatchMatmulAttrs const &attrs,
    ParallelTensorDimDegrees const &lhs,
    ParallelTensorDimDegrees const &rhs) {

  num_ptensor_shard_dims_t num_tensor_dims =
      require_same(get_ptensor_dim_degrees_num_shard_dims(lhs),
                   get_ptensor_dim_degrees_num_shard_dims(rhs));

  parallel_tensor_dim_idx_t row_dim =
      shard_dim_idx_for_relative(-2, num_tensor_dims);
  parallel_tensor_dim_idx_t col_dim =
      shard_dim_idx_for_relative(-1, num_tensor_dims);

  FFOrdered<positive_int> leading_degrees = require_same(
      ff_ordered_slice(
          lhs.shard_degrees, relative_ff_dim_t{0}, relative_ff_dim_t{-2}),
      ff_ordered_slice(
          rhs.shard_degrees, relative_ff_dim_t{0}, relative_ff_dim_t{-2}));

  positive_int reduction_parallelism_degree =
      require_same(get_degree_for_parallel_tensor_dim_idx(lhs, col_dim),
                   get_degree_for_parallel_tensor_dim_idx(rhs, row_dim));

  positive_int lhs_row_degree =
      get_degree_for_parallel_tensor_dim_idx(lhs, row_dim);

  positive_int rhs_column_degree =
      get_degree_for_parallel_tensor_dim_idx(rhs, col_dim);

  ASSERT(lhs_row_degree * lhs.sum_degree.value ==
         rhs.discard_copy_degree.value);

  ASSERT(rhs_column_degree * rhs.sum_degree.value ==
         lhs.discard_copy_degree.value);

  return ParallelTensorDimDegrees{
      /*sum_degree=*/SumDegree{
          reduction_parallelism_degree * lhs.sum_degree.value *
              rhs.sum_degree.value,
      },
      /*discard_copy_degree=*/DiscardCopyDegree{1_p},
      /*shard_degrees=*/
      ff_ordered_concat(leading_degrees,
                        FFOrdered<positive_int>{
                            lhs_row_degree,
                            rhs_column_degree,
                        }),
  };
}

ParallelTensorShape
    batch_matmul_get_output_parallel_shape(BatchMatmulAttrs const &attrs,
                                           ParallelTensorShape const &lhs,
                                           ParallelTensorShape const &rhs) {
  TensorShape output_shape = batch_matmul_get_output_shape(
      attrs, get_reduced_shape(lhs), get_reduced_shape(rhs));

  ParallelTensorDimDegrees output_degrees =
      batch_matmul_get_output_parallel_dim_degrees(
          attrs, get_parallel_degrees(lhs), get_parallel_degrees(rhs));

  return lift_shape_to_parallel_with_degrees(output_shape, output_degrees);
}

StandardOperatorTaskGroup batch_matmul_get_task_group(
    BatchMatmulAttrs const &attrs,
    ParallelTensorDimDegrees const &lhs_input_degrees,
    ParallelTensorDimDegrees const &rhs_input_degrees) {
  num_ptensor_shard_dims_t input_num_shard_dims =
      require_same(get_ptensor_dim_degrees_num_shard_dims(lhs_input_degrees),
                   get_ptensor_dim_degrees_num_shard_dims(rhs_input_degrees));

  ParallelTensorDimDegrees output_degrees =
      batch_matmul_get_output_parallel_dim_degrees(
          attrs, lhs_input_degrees, rhs_input_degrees);

  return StandardOperatorTaskGroup{
      set_filtrans(
          binary_cartesian_product(
              get_parallel_tensor_space_coordinates(lhs_input_degrees),
              get_parallel_tensor_space_coordinates(rhs_input_degrees)),
          [&](std::pair<ParallelTensorSpaceCoordinate,
                        ParallelTensorSpaceCoordinate> const &coords)
              -> std::optional<AbstractedOperatorAtomicTaskShardBinding> {
            ParallelTensorSpaceCoordinate lhs_input_coord = coords.first;
            ParallelTensorSpaceCoordinate rhs_input_coord = coords.second;

            parallel_tensor_dim_idx_t sum_dim = sum_dim_idx();
            parallel_tensor_dim_idx_t discard_copy_dim = discard_copy_dim_idx();

            std::set<parallel_tensor_dim_idx_t> leading_dims =
                shard_dim_idxs_for_exclusive_interval(
                    0, -2, input_num_shard_dims);

            parallel_tensor_dim_idx_t row_dim =
                shard_dim_idx_for_relative(-2, input_num_shard_dims);

            parallel_tensor_dim_idx_t col_dim =
                shard_dim_idx_for_relative(-1, input_num_shard_dims);

            // c
            OrthotopeBoundedCoord lhs_leading_dims_coord =
                orthotope_bounded_coord_for_ptensor_dims(
                    lhs_input_degrees, lhs_input_coord, leading_dims);

            OrthotopeBoundedCoord rhs_leading_dims_coord =
                orthotope_bounded_coord_for_ptensor_dims(
                    rhs_input_degrees, rhs_input_coord, leading_dims);

            if (lhs_leading_dims_coord != rhs_leading_dims_coord) {
              return std::nullopt;
            }

            OrthotopeBoundedCoord data_parallelism_coord =
                require_same(lhs_leading_dims_coord, rhs_leading_dims_coord);

            // h
            BoundedComponent output_column_parallelism_coord =
                bounded_component_for_ptensor_dim(
                    rhs_input_degrees, rhs_input_coord, col_dim);

            // d
            BoundedComponent output_row_parallelism_coord =
                bounded_component_for_ptensor_dim(
                    lhs_input_degrees, lhs_input_coord, row_dim);

            // e
            BoundedComponent lhs_reduction_parallelism =
                bounded_component_for_ptensor_dim(
                    lhs_input_degrees, lhs_input_coord, col_dim);

            BoundedComponent rhs_reduction_parallelism =
                bounded_component_for_ptensor_dim(
                    rhs_input_degrees, rhs_input_coord, row_dim);

            if (lhs_reduction_parallelism != rhs_reduction_parallelism) {
              return std::nullopt;
            }

            BoundedComponent reduction_parallelism = require_same(
                lhs_reduction_parallelism, rhs_reduction_parallelism);

            // a
            BoundedComponent lhs_preexisting_sum_parallelism_coord =
                bounded_component_for_ptensor_dim(
                    lhs_input_degrees, lhs_input_coord, sum_dim);

            // f
            BoundedComponent rhs_preexisting_sum_parallelism_coord =
                bounded_component_for_ptensor_dim(
                    rhs_input_degrees, rhs_input_coord, sum_dim);

            BoundedComponent output_sum_component = assert_unwrap(
                flatten_orthotope_bounded_coord(make_3d_orthotope_bounded_coord(
                    reduction_parallelism,
                    lhs_preexisting_sum_parallelism_coord,
                    rhs_preexisting_sum_parallelism_coord)));

            BoundedComponent output_copy_component =
                trivial_bounded_component();

            OrthotopeBoundedCoord output_shard_components =
                orthotope_bounded_coord_product(
                    data_parallelism_coord,
                    lift_bounded_component(output_row_parallelism_coord),
                    lift_bounded_component(output_column_parallelism_coord));

            ParallelTensorSpaceCoordinate output_coord =
                parallel_tensor_space_coordinate_from_bounded_orthotope_components(
                    /*sum_degree=*/output_sum_component,
                    /*discard_copy_degree=*/output_copy_component,
                    /*shard_coords=*/output_shard_components);

            return AbstractedOperatorAtomicTaskShardBinding{
                /*tensor_corods=*/{
                    {
                        TensorSlotName::LHS_INPUT,
                        lhs_input_coord,
                    },
                    {
                        TensorSlotName::RHS_INPUT,
                        rhs_input_coord,
                    },
                    {
                        TensorSlotName::OUTPUT,
                        output_coord,
                    },
                },
                /*task_coord=*/
                task_coord_matching_parallel_tensor_space_coordinate(
                    output_coord, output_degrees),
            };
          }),
  };
}

} // namespace FlexFlow
