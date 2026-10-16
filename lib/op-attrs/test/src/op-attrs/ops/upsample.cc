#include "op-attrs/ops/upsample.h"
#include "kernels/accessor.h"
#include "kernels/create_zero_filled_accessor.h"
#include "kernels/emulated_parallel_tensor/shard_signature_instance_is_valid.h"
#include "kernels/local_cpu_allocator.h"
#include "kernels/upsample_kernels_cpu.h"
#include "op-attrs/parallel_tensor_dims.h"
#include "op-attrs/parallel_tensor_shape.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("upsample_get_task_group") {
    Allocator cpu_allocator = create_local_cpu_memory_allocator();

    auto mk_degrees = [](int sum_degree,
                         int discard_copy_degree,
                         int batch_dim_degree,
                         int channel_dim_degree,
                         int height_dim_degree,
                         int width_dim_degree) -> ParallelTensorDimDegrees {
      return ParallelTensorDimDegrees{
          /*sum_degree=*/SumDegree{positive_int{sum_degree}},
          /*discard_copy_degree=*/
          DiscardCopyDegree{positive_int{discard_copy_degree}},
          /*shard_degrees=*/
          FFOrdered<positive_int>{
              positive_int{batch_dim_degree},
              positive_int{channel_dim_degree},
              positive_int{height_dim_degree},
              positive_int{width_dim_degree},
          },
      };
    };

    TensorShape input_shard_shape = TensorShape{
        /*dims=*/TensorDims{
            FFOrdered<positive_int>{
                4_p,
                2_p,
                6_p,
                6_p,
            },
        },
        /*data_type=*/DataType::FLOAT,
    };

    UpsampleAttrs attrs = UpsampleAttrs{
        /*scale_factor=*/3_ge2,
        /*mode=*/UpsampleMode::NEAREST,
    };

    auto run_upsample =
        [&](std::map<TensorSlotName, GenericTensorAccessorR> const
                &input_shards)
        -> std::map<TensorSlotName, GenericTensorAccessorR> {
      GenericTensorAccessorR input_shard =
          input_shards.at(TensorSlotName::INPUT);

      TensorShape output_shard_shape =
          upsample_get_output_shape(attrs, input_shard.shape);
      GenericTensorAccessorW output_shard =
          create_zero_filled_accessor_w(output_shard_shape, cpu_allocator);

      upsample_cpu_forward_kernel(
          /*attrs=*/attrs,
          /*input=*/input_shard,
          /*output=*/output_shard);

      return std::map<TensorSlotName, GenericTensorAccessorR>{
          {
              TensorSlotName::OUTPUT,
              read_only_accessor_from_write_accessor(output_shard),
          },
      };
    };

    auto upsample_shard_signature_instance_is_valid =
        [&](ParallelTensorDimDegrees const &input_degrees) -> bool {
      ParallelTensorShape input_shape =
          lift_shape_to_parallel_with_degrees(input_shard_shape, input_degrees);

      std::map<TensorSlotName, ParallelTensorShape> input_shapes = {
          {
              TensorSlotName::INPUT,
              input_shape,
          },
      };

      return shard_signature_instance_is_valid(
          /*attrs=*/ComputationGraphOpAttrs{attrs},
          /*input_shapes=*/input_shapes,
          /*run_op=*/run_upsample,
          /*seed=*/0);
    };

    SUBCASE("preexisting sum parallelism") {
      ParallelTensorDimDegrees input_degrees = mk_degrees(3, 1, 1, 1, 1, 1);

      CHECK(upsample_shard_signature_instance_is_valid(input_degrees));
    }

    SUBCASE("data parallelism") {
      ParallelTensorDimDegrees input_degrees = mk_degrees(1, 1, 2, 1, 1, 1);

      CHECK(upsample_shard_signature_instance_is_valid(input_degrees));
    }

    SUBCASE("channel parallelism parallelism") {
      ParallelTensorDimDegrees input_degrees = mk_degrees(1, 1, 1, 2, 1, 1);

      CHECK(upsample_shard_signature_instance_is_valid(input_degrees));
    }

    SUBCASE("height parallelism") {
      ParallelTensorDimDegrees input_degrees = mk_degrees(1, 1, 1, 1, 2, 1);

      CHECK(upsample_shard_signature_instance_is_valid(input_degrees));
    }

    SUBCASE("height parallelism") {
      ParallelTensorDimDegrees input_degrees = mk_degrees(1, 1, 1, 1, 2, 1);

      CHECK(upsample_shard_signature_instance_is_valid(input_degrees));
    }

    SUBCASE("width parallelism") {
      ParallelTensorDimDegrees input_degrees = mk_degrees(1, 1, 1, 1, 1, 2);

      CHECK(upsample_shard_signature_instance_is_valid(input_degrees));
    }
  }
}
