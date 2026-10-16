#include "op-attrs/ops/flat.h"
#include "kernels/accessor.h"
#include "kernels/create_zero_filled_accessor.h"
#include "kernels/emulated_parallel_tensor/shard_signature_instance_is_valid.h"
#include "kernels/flat_kernels_cpu.h"
#include "kernels/local_cpu_allocator.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "utils/expected.h"
#include "utils/fmt/expected.h"
#include "utils/fmt/optional.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("flat_get_output_shape") {
    TensorShape input_shape = TensorShape{
        TensorDims{FFOrdered{
            2_p,
            4_p,
            2_p,
            3_p,
        }},
        DataType::FLOAT,
    };

    SUBCASE("flatten all dims") {
      FlatAttrs attrs = FlatAttrs{
          /*start_dim=*/ff_dim_t{0_n},
          /*end_dim=*/ff_dim_t{3_n},
      };

      TensorShape result = flat_get_output_shape(attrs, input_shape);
      TensorShape correct = TensorShape{
          TensorDims{FFOrdered{
              2_p * 4_p * 2_p * 3_p,
          }},
          DataType::FLOAT,
      };

      CHECK(result == correct);
    }

    SUBCASE("flatten trailing dims") {
      FlatAttrs attrs = FlatAttrs{
          /*start_dim=*/ff_dim_t{2_n},
          /*end_dim=*/ff_dim_t{3_n},
      };

      TensorShape result = flat_get_output_shape(attrs, input_shape);
      TensorShape correct = TensorShape{
          TensorDims{FFOrdered{
              2_p,
              4_p,
              2_p * 3_p,
          }},
          DataType::FLOAT,
      };

      CHECK(result == correct);
    }

    SUBCASE("flatten leading dims") {
      FlatAttrs attrs = FlatAttrs{
          /*start_dim=*/ff_dim_t{0_n},
          /*end_dim=*/ff_dim_t{1_n},
      };

      TensorShape result = flat_get_output_shape(attrs, input_shape);
      TensorShape correct = TensorShape{
          TensorDims{FFOrdered{
              2_p * 4_p,
              2_p,
              3_p,
          }},
          DataType::FLOAT,
      };

      CHECK(result == correct);
    }

    SUBCASE("flatten middle dims") {
      FlatAttrs attrs = FlatAttrs{
          /*start_dim=*/ff_dim_t{1_n},
          /*end_dim=*/ff_dim_t{2_n},
      };

      TensorShape result = flat_get_output_shape(attrs, input_shape);
      TensorShape correct = TensorShape{
          TensorDims{FFOrdered{
              2_p,
              4_p * 2_p,
              3_p,
          }},
          DataType::FLOAT,
      };

      CHECK(result == correct);
    }

    SUBCASE("flatten single dims (start_dim == end_dim)") {
      FlatAttrs attrs = FlatAttrs{
          /*start_dim=*/ff_dim_t{2_n},
          /*end_dim=*/ff_dim_t{2_n},
      };

      TensorShape result = flat_get_output_shape(attrs, input_shape);
      TensorShape correct = input_shape;

      CHECK(result == correct);
    }

    SUBCASE("flatten no dims (start_dim > end_dim)") {
      FlatAttrs attrs = FlatAttrs{
          /*start_dim=*/ff_dim_t{3_n},
          /*end_dim=*/ff_dim_t{2_n},
      };

      TensorShape result = flat_get_output_shape(attrs, input_shape);
      TensorShape correct = input_shape;

      CHECK(result == correct);
    }
  }

  TEST_CASE("flat_get_output_parallel_dim_degrees") {
    FlatAttrs attrs = FlatAttrs{/*start_dim=*/ff_dim_t{1_n},
                                /*end_dim=*/ff_dim_t{2_n}};

    SUBCASE("allows shard parallelism in non-flattened dims") {
      ParallelTensorDimDegrees input = ParallelTensorDimDegrees{
          SumDegree{1_p},
          DiscardCopyDegree{1_p},
          FFOrdered{2_p, 1_p, 1_p, 3_p},
      };

      ParallelTensorDimDegrees result =
          flat_get_output_parallel_dim_degrees(attrs, input);
      ParallelTensorDimDegrees correct = ParallelTensorDimDegrees{
          SumDegree{1_p},
          DiscardCopyDegree{1_p},
          FFOrdered{2_p, 1_p, 3_p},
      };

      CHECK(result == correct);
    }

    SUBCASE("does not allow shard parallelism in flattened dims") {
      ParallelTensorDimDegrees input = ParallelTensorDimDegrees{
          SumDegree{1_p},
          DiscardCopyDegree{1_p},
          FFOrdered{1_p, 1_p, 2_p, 1_p},
      };

      CHECK_THROWS(flat_get_output_parallel_dim_degrees(attrs, input));
    }

    SUBCASE("allows sum parallelism") {
      ParallelTensorDimDegrees input = ParallelTensorDimDegrees{
          SumDegree{2_p},
          DiscardCopyDegree{1_p},
          FFOrdered{1_p, 1_p, 1_p, 1_p},
      };

      ParallelTensorDimDegrees result =
          flat_get_output_parallel_dim_degrees(attrs, input);
      ParallelTensorDimDegrees correct = ParallelTensorDimDegrees{
          SumDegree{2_p},
          DiscardCopyDegree{1_p},
          FFOrdered{1_p, 1_p, 1_p},
      };

      CHECK(result == correct);
    }

    SUBCASE("allows discard copy parallelism") {
      ParallelTensorDimDegrees input = ParallelTensorDimDegrees{
          SumDegree{1_p},
          DiscardCopyDegree{2_p},
          FFOrdered{1_p, 1_p, 1_p, 1_p},
      };

      ParallelTensorDimDegrees result =
          flat_get_output_parallel_dim_degrees(attrs, input);
      ParallelTensorDimDegrees correct = ParallelTensorDimDegrees{
          SumDegree{1_p},
          DiscardCopyDegree{2_p},
          FFOrdered{1_p, 1_p, 1_p},
      };

      CHECK(result == correct);
    }
  }

  TEST_CASE("flat_get_output_parallel_shape") {
    // since most of the edge cases are already tested in
    // flat_get_output_shape and flat_get_output_parallel_dim_degrees,
    // here we just do a basic check that they compose

    ParallelTensorShape input_shape = ParallelTensorShape{
        ParallelTensorDims{
            FFOrdered<ShardParallelDim>{
                ShardParallelDim{4_p, 2_p},
                ShardParallelDim{8_p, 1_p},
                ShardParallelDim{6_p, 1_p},
                ShardParallelDim{9_p, 3_p},
            },
            ReplicaParallelDimSet{
                SumDegree{7_p},
                DiscardCopyDegree{5_p},
            },
        },
        DataType::FLOAT,
    };

    FlatAttrs attrs = FlatAttrs{
        /*start_dim=*/ff_dim_t{nonnegative_int{1_p}},
        /*end_dim=*/ff_dim_t{nonnegative_int{2_p}},
    };

    ParallelTensorShape result =
        flat_get_output_parallel_shape(attrs, input_shape);
    ParallelTensorShape correct = ParallelTensorShape{
        ParallelTensorDims{
            FFOrdered<ShardParallelDim>{
                ShardParallelDim{4_p, 2_p},
                ShardParallelDim{8_p * 6_p, 1_p},
                ShardParallelDim{9_p, 3_p},
            },
            ReplicaParallelDimSet{
                SumDegree{7_p},
                DiscardCopyDegree{5_p},
            },
        },
        DataType::FLOAT,
    };

    CHECK(result == correct);
  }

  TEST_CASE("flat_get_task_group") {
    Allocator cpu_allocator = create_local_cpu_memory_allocator();

    FlatAttrs attrs = FlatAttrs{
        /*start_dim=*/ff_dim_t{1_n},
        /*end_dim=*/ff_dim_t{2_n},
    };

    TensorShape input_shape = TensorShape{
        TensorDims{
            FFOrdered{
                8_p,
                6_p,
                4_p,
                5_p,
            },
        },
        DataType::FLOAT,
    };

    auto mk_dim_degrees =
        [&](int sum_degree,
            int discard_copy_degree,
            int dim0_shard_degree,
            int dim1_shard_degree,
            int dim2_shard_degree,
            int dim3_shard_degree) -> ParallelTensorDimDegrees {
      return ParallelTensorDimDegrees{
          /*sum_degree=*/SumDegree{positive_int{sum_degree}},
          /*discard_copy_degree=*/
          DiscardCopyDegree{positive_int{discard_copy_degree}},
          /*shard_degrees=*/
          FFOrdered{
              positive_int{dim0_shard_degree},
              positive_int{dim1_shard_degree},
              positive_int{dim2_shard_degree},
              positive_int{dim3_shard_degree},
          },
      };
    };

    auto run_flat = [&](std::map<TensorSlotName, GenericTensorAccessorR> const
                            &incoming_shards)
        -> std::map<TensorSlotName, GenericTensorAccessorR> {
      GenericTensorAccessorR input_shard =
          incoming_shards.at(TensorSlotName::INPUT);
      TensorShape output_shard_shape = flat_get_output_shape(
          attrs, get_tensor_shape_for_accessor_r(input_shard));
      GenericTensorAccessorW output_shard =
          create_zero_filled_accessor_w(output_shard_shape, cpu_allocator);

      flat_cpu_forward_kernel(
          /*input=*/input_shard,
          /*output=*/output_shard);

      return std::map<TensorSlotName, GenericTensorAccessorR>{
          {
              TensorSlotName::OUTPUT,
              read_only_accessor_from_write_accessor(output_shard),
          },
      };
    };

    auto flat_shard_signature_instance_is_valid =
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
          /*run_op=*/run_flat,
          /*seed=*/0);
    };

    SUBCASE("data parallelism") {
      ParallelTensorDimDegrees input_dim_degrees =
          mk_dim_degrees(1, 1, 2, 1, 1, 1);

      CHECK(flat_shard_signature_instance_is_valid(input_dim_degrees));
    }
  }
}
