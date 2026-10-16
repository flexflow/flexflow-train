#include "op-attrs/ops/reduce.h"
#include "op-attrs/ff_ordered/ff_ordered_filtrans_with_idx.h"
#include "op-attrs/parallel_tensor_dim_degrees.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "op-attrs/task_space_coordinate.h"
#include "op-attrs/tensor_dims.h"
#include "op-attrs/tensor_shape.h"
#include "utils/containers/are_disjoint.h"
#include "utils/containers/contains.h"
#include "utils/containers/filtrans.h"
#include "utils/containers/is_subseteq_of.h"

namespace FlexFlow {

TensorShape reduce_get_output_shape(ReduceAttrs const &attrs,
                                    TensorShape const &input_shape) {
  {
    std::set<ff_dim_t> input_dims_set = get_ff_dim_t_set(input_shape.dims);
    ASSERT(is_subseteq_of(attrs.axes.unwrap_as_set(), input_dims_set));
  }

  if (attrs.keepdims) {
    return input_shape;
  } else {
    return tensor_shape_drop_dims(input_shape, [&](ff_dim_t d) -> bool {
      return contains(attrs.axes, d);
    });
  }
}

ParallelTensorDimDegrees reduce_get_output_parallel_dim_degrees(
    ReduceAttrs const &attrs, ParallelTensorDimDegrees const &input_degrees) {

  auto only_shard_dims =
      [&](std::set<parallel_tensor_dim_idx_t> const &ds) -> std::set<ff_dim_t> {
    return filtrans(
        ds, [&](parallel_tensor_dim_idx_t const &d) -> std::optional<ff_dim_t> {
          return d.try_require_shard_dim();
        });
  };

  std::set<ff_dim_t> all_shard_dims =
      only_shard_dims(get_parallel_tensor_dim_indices(input_degrees));

  ASSERT(is_subseteq_of(attrs.axes.unwrap_as_set(), all_shard_dims));

  std::set<ff_dim_t> nontrivial_shard_dims = only_shard_dims(
      get_nontrivial_parallel_tensor_dim_indices(input_degrees));

  ASSERT(are_disjoint(attrs.axes.unwrap_as_set(), nontrivial_shard_dims),
         attrs.axes,
         nontrivial_shard_dims);

  switch (attrs.op) {
    case ReduceOp::SUM: {
      break;
    }
    case ReduceOp::AVG: {
      ASSERT(input_degrees.sum_degree.value == 1);
      break;
    }
    default: {
      PANIC(fmt::format("Unexpected ReduceOp {}", attrs.op));
    }
  }

  if (attrs.keepdims) {
    return input_degrees;
  } else {
    return parallel_dim_degrees_filter_shard_dims(
        input_degrees,
        [&](ff_dim_t d) -> bool { return !contains(attrs.axes, d); });
  }
}

ParallelTensorShape
    reduce_get_output_parallel_shape(ReduceAttrs const &attrs,
                                     ParallelTensorShape const &input_shape) {
  TensorShape output_shape =
      reduce_get_output_shape(attrs, get_reduced_shape(input_shape));

  ParallelTensorDimDegrees output_degrees =
      reduce_get_output_parallel_dim_degrees(attrs,
                                             get_parallel_degrees(input_shape));

  return lift_shape_to_parallel_with_degrees(output_shape, output_degrees);
}

StandardOperatorTaskGroup
    reduce_get_task_group(ReduceAttrs const &attrs,
                          ParallelTensorDimDegrees const &input_degrees) {
  ParallelTensorDimDegrees output_degrees =
      reduce_get_output_parallel_dim_degrees(attrs, input_degrees);

  auto output_coord_from_input_coord =
      [&](ParallelTensorSpaceCoordinate const &input_coord)
      -> ParallelTensorSpaceCoordinate {
    return ParallelTensorSpaceCoordinate{
        /*sum_component=*/input_coord.sum_component,
        /*discard_copy_component=*/input_coord.discard_copy_component,
        /*shard_components=*/
        ff_ordered_filtrans_with_idx(
            input_coord.shard_components,
            [&](ff_dim_t idx,
                nonnegative_int component) -> std::optional<nonnegative_int> {
              if (!contains(attrs.axes, idx)) {
                return component;
              }

              if (attrs.keepdims) {
                return 0_n;
              } else {
                return std::nullopt;
              }
            }),
    };
  };

  return StandardOperatorTaskGroup{
      transform(get_parallel_tensor_space_coordinates(input_degrees),
                [&](ParallelTensorSpaceCoordinate const &input_coord)
                    -> AbstractedOperatorAtomicTaskShardBinding {
                  ParallelTensorSpaceCoordinate output_coord =
                      output_coord_from_input_coord(input_coord);

                  return AbstractedOperatorAtomicTaskShardBinding{
                      /*tensor_corods=*/{
                          {
                              TensorSlotName::INPUT,
                              input_coord,
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
