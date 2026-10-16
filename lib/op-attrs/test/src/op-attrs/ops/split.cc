#include "op-attrs/ops/split.h"
#include "kernels/create_zero_filled_accessor.h"
#include "kernels/emulated_parallel_tensor/shard_signature_instance_is_valid.h"
#include "kernels/local_cpu_allocator.h"
#include "kernels/split_kernels_cpu.h"
#include "op-attrs/parallel_tensor_dims.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "utils/containers/map_from_keys_and_values.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("split_get_output_shapes") {
    TensorShape input_shape = TensorShape{
        TensorDims{FFOrdered{4_p, 3_p, 10_p, 10_p}},
        DataType::FLOAT,
    };

    SUBCASE("splits are too small for input") {
      SplitAttrs attrs = SplitAttrs{
          /*splits=*/std::vector<positive_int>{
              2_p,
              4_p,
              3_p,
          },
          /*axis=*/ff_dim_t{2_n},
      };

      CHECK_THROWS(split_get_output_shapes(attrs, input_shape));
    }

    SUBCASE("splits are too large for input") {
      SplitAttrs attrs = SplitAttrs{
          /*splits=*/std::vector<positive_int>{
              2_p,
              6_p,
              3_p,
          },
          /*axis=*/ff_dim_t{2_n},
      };

      CHECK_THROWS(split_get_output_shapes(attrs, input_shape));
    }

    SUBCASE("axis does not exist in input") {
      SplitAttrs attrs = SplitAttrs{
          /*splits=*/std::vector<positive_int>{
              2_p,
              5_p,
              3_p,
          },
          /*axis=*/ff_dim_t{4_n},
      };

      CHECK_THROWS(split_get_output_shapes(attrs, input_shape));
    }

    SUBCASE("correct usage") {
      SplitAttrs attrs = SplitAttrs{
          /*splits=*/std::vector<positive_int>{
              2_p,
              5_p,
              3_p,
          },
          /*axis=*/ff_dim_t{2_n},
      };

      std::vector<TensorShape> result =
          split_get_output_shapes(attrs, input_shape);

      auto mk_correct_shape = [&](positive_int x) -> TensorShape {
        return TensorShape{
            TensorDims{FFOrdered{4_p, 3_p, x, 10_p}},
            DataType::FLOAT,
        };
      };

      std::vector<TensorShape> correct = {
          mk_correct_shape(2_p),
          mk_correct_shape(5_p),
          mk_correct_shape(3_p),
      };

      CHECK(result == correct);
    }
  }

  TEST_CASE("split_get_output_parallel_dim_degrees") {
    SplitAttrs attrs = SplitAttrs{
        /*splits=*/std::vector<positive_int>{
            3_p,
            2_p,
            5_p,
        },
        /*axis=*/ff_dim_t{2_n},
    };

    SUBCASE("split degree is 1") {
      ParallelTensorDimDegrees input_dim_degrees = ParallelTensorDimDegrees{
          /*sum_degree=*/SumDegree{2_p},
          /*discard_copy_degree=*/DiscardCopyDegree{1_p},
          /*shard_degrees=*/
          FFOrdered<positive_int>{
              2_p,
              1_p,
              1_p,
              1_p,
          },
      };

      std::vector<ParallelTensorDimDegrees> result =
          split_get_output_parallel_dim_degrees(attrs, input_dim_degrees);
      std::vector<ParallelTensorDimDegrees> correct = {
          input_dim_degrees,
          input_dim_degrees,
          input_dim_degrees,
      };

      CHECK(result == correct);
    }

    SUBCASE("split degree is not 1") {
      ParallelTensorDimDegrees input_dim_degrees = ParallelTensorDimDegrees{
          /*sum_degree=*/SumDegree{1_p},
          /*discard_copy_degree=*/DiscardCopyDegree{1_p},
          /*shard_degrees=*/
          FFOrdered<positive_int>{
              1_p,
              1_p,
              2_p,
              1_p,
          },
      };

      CHECK_THROWS(
          split_get_output_parallel_dim_degrees(attrs, input_dim_degrees));
    }
  }

  TEST_CASE("split_get_task_group") {
    Allocator cpu_allocator = create_local_cpu_memory_allocator();

    SplitAttrs attrs = SplitAttrs{
        /*splits=*/{3_p, 8_p, 2_p},
        /*axis=*/ff_dim_t{1_n},
    };

    TensorShape input_shape = TensorShape{
        TensorDims{
            FFOrdered{
                6_p,
                13_p,
                4_p,
            },
        },
        DataType::FLOAT,
    };

    auto mk_dim_degrees =
        [&](int sum_degree,
            int discard_copy_degree,
            int dim0_shard_degree,
            int dim1_shard_degree,
            int dim2_shard_degree) -> ParallelTensorDimDegrees {
      return ParallelTensorDimDegrees{
          /*sum_degree=*/SumDegree{positive_int{sum_degree}},
          /*discard_copy_degree=*/
          DiscardCopyDegree{positive_int{discard_copy_degree}},
          /*shard_degrees=*/
          FFOrdered{
              positive_int{dim0_shard_degree},
              positive_int{dim1_shard_degree},
              positive_int{dim2_shard_degree},
          },
      };
    };

    auto run_split = [&](std::map<TensorSlotName, GenericTensorAccessorR> const
                             &incoming_shards)
        -> std::map<TensorSlotName, GenericTensorAccessorR> {
      GenericTensorAccessorR input_shard =
          incoming_shards.at(TensorSlotName::INPUT);
      std::vector<TensorShape> output_shard_shapes = split_get_output_shapes(
          attrs, get_tensor_shape_for_accessor_r(input_shard));
      std::vector<GenericTensorAccessorW> output_shards = transform(
          output_shard_shapes,
          [&](TensorShape const &output_shard_shape) -> GenericTensorAccessorW {
            return create_zero_filled_accessor_w(output_shard_shape,
                                                 cpu_allocator);
          });

      split_cpu_forward_kernel(
          /*attrs=*/attrs,
          /*input=*/input_shard,
          /*outputs=*/output_shards);

      return map_from_keys_and_values(
          split_get_output_slot_names(attrs),
          transform(
              output_shards,
              [&](GenericTensorAccessorW const &t) -> GenericTensorAccessorR {
                return read_only_accessor_from_write_accessor(t);
              }));
    };

    auto split_shard_signature_instance_is_valid =
        [&](ParallelTensorDimDegrees const &input_degrees) -> bool {
      ParallelTensorShape input_parallel_shape =
          lift_shape_to_parallel_with_degrees(input_shape, input_degrees);

      std::map<TensorSlotName, ParallelTensorShape> input_shapes = {
          {
              TensorSlotName::INPUT,
              input_parallel_shape,
          },
      };

      return shard_signature_instance_is_valid(
          /*attrs=*/ComputationGraphOpAttrs{attrs},
          /*input_shapes=*/input_shapes,
          /*run_op=*/run_split,
          /*seed=*/0);
    };

    SUBCASE("data parallelism") {
      ParallelTensorDimDegrees input_dim_degrees =
          mk_dim_degrees(1, 1, 2, 1, 1);

      CHECK(split_shard_signature_instance_is_valid(input_dim_degrees));
    }

    SUBCASE("mixed parallelism") {
      ParallelTensorDimDegrees input_dim_degrees =
          mk_dim_degrees(3, 2, 3, 1, 2);

      CHECK(split_shard_signature_instance_is_valid(input_dim_degrees));
    }
  }
}
