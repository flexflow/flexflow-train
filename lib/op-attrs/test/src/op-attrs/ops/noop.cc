#include "op-attrs/ops/noop.h"
#include "kernels/accessor.h"
#include "kernels/create_zero_filled_accessor.h"
#include "kernels/emulated_parallel_tensor/shard_signature_instance_is_valid.h"
#include "kernels/flat_kernels_cpu.h"
#include "kernels/local_cpu_allocator.h"
#include "op-attrs/parallel_tensor_shape.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("noop_get_output_shape") {
    TensorShape input = TensorShape{
        TensorDims{
            FFOrdered<positive_int>{
                12_p,
                8_p,
                13_p,
            },
        },
        DataType::FLOAT,
    };

    NoopAttrs attrs = NoopAttrs{};

    TensorShape result = noop_get_output_shape(attrs, input);
    TensorShape correct = input;

    CHECK(result == correct);
  }

  TEST_CASE("noop_get_output_parallel_dim_degrees") {
    ParallelTensorDimDegrees input = ParallelTensorDimDegrees{
        SumDegree{3_p},
        DiscardCopyDegree{5_p},
        FFOrdered<positive_int>{
            6_p,
            4_p,
        },
    };

    NoopAttrs attrs = NoopAttrs{};

    ParallelTensorDimDegrees result =
        noop_get_output_parallel_dim_degrees(attrs, input);
    ParallelTensorDimDegrees correct = input;

    CHECK(result == correct);
  }

  TEST_CASE("noop_get_output_parallel_shape") {
    ParallelTensorShape input = ParallelTensorShape{
        ParallelTensorDims{
            FFOrdered<ShardParallelDim>{
                ShardParallelDim{12_p, 3_p},
                ShardParallelDim{8_p, 2_p},
                ShardParallelDim{13_p, 1_p},
            },
            ReplicaParallelDimSet{
                SumDegree{5_p},
                DiscardCopyDegree{7_p},
            },
        },
        DataType::FLOAT,
    };

    NoopAttrs attrs = NoopAttrs{};

    ParallelTensorShape result = noop_get_output_parallel_shape(attrs, input);
    ParallelTensorShape correct = input;

    CHECK(result == correct);
  }

  TEST_CASE("noop_get_task_group") {
    Allocator cpu_allocator = create_local_cpu_memory_allocator();

    auto mk_degrees =
        [](int sum_degree,
           int discard_copy_degree,
           int batch_dim_degree,
           int inner_dimension_1_degree,
           int inner_dimension_2_degree) -> ParallelTensorDimDegrees {
      return ParallelTensorDimDegrees{
          /*sum_degree=*/SumDegree{positive_int{sum_degree}},
          /*discard_copy_degree=*/
          DiscardCopyDegree{positive_int{discard_copy_degree}},
          /*shard_degrees=*/
          FFOrdered<positive_int>{
              positive_int{batch_dim_degree},
              positive_int{inner_dimension_1_degree},
              positive_int{inner_dimension_2_degree},
          },
      };
    };

    TensorShape input_shard_shape = TensorShape{
        /*dims=*/TensorDims{
            FFOrdered<positive_int>{
                4_p,
                2_p,
                6_p,
            },
        },
        /*data_type=*/DataType::FLOAT,
    };

    NoopAttrs attrs = NoopAttrs{};

    auto run_noop = [&](std::map<TensorSlotName, GenericTensorAccessorR> const
                            &input_shards)
        -> std::map<TensorSlotName, GenericTensorAccessorR> {
      GenericTensorAccessorR input_shard =
          input_shards.at(TensorSlotName::INPUT);
      TensorShape output_shard_shape =
          noop_get_output_shape(attrs, input_shard.shape);
      GenericTensorAccessorW output_shard =
          create_zero_filled_accessor_w(output_shard_shape, cpu_allocator);

      copy_accessor_data_to_l_from_r_strict(output_shard, input_shard);

      return std::map<TensorSlotName, GenericTensorAccessorR>{
          {
              TensorSlotName::OUTPUT,
              read_only_accessor_from_write_accessor(output_shard),
          },
      };
    };

    auto noop_shard_signature_instance_is_valid =
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
          /*run_op=*/run_noop,
          /*seed=*/0);
    };

    SUBCASE("sum parallelism") {
      ParallelTensorDimDegrees input_degrees = mk_degrees(2, 1, 1, 1, 1);

      CHECK(noop_shard_signature_instance_is_valid(input_degrees));
    }

    SUBCASE("discard copy parallelism") {
      ParallelTensorDimDegrees input_degrees = mk_degrees(1, 2, 1, 1, 1);

      CHECK(noop_shard_signature_instance_is_valid(input_degrees));
    }

    SUBCASE("data parallelism") {
      ParallelTensorDimDegrees input_degrees = mk_degrees(1, 1, 2, 1, 1);

      CHECK(noop_shard_signature_instance_is_valid(input_degrees));
    }

    SUBCASE("inner dimension 1 parallelism") {
      ParallelTensorDimDegrees input_degrees = mk_degrees(1, 1, 1, 2, 1);

      CHECK(noop_shard_signature_instance_is_valid(input_degrees));
    }

    SUBCASE("inner dimension 2 parallelism") {
      ParallelTensorDimDegrees input_degrees = mk_degrees(1, 1, 1, 1, 2);

      CHECK(noop_shard_signature_instance_is_valid(input_degrees));
    }

    SUBCASE("hybrid parallelism") {
      ParallelTensorDimDegrees input_degrees = mk_degrees(2, 2, 2, 2, 2);

      CHECK(noop_shard_signature_instance_is_valid(input_degrees));
    }
  }
}
