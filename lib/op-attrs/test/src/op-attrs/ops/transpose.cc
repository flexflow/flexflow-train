#include "op-attrs/ops/transpose.h"
#include "kernels/create_zero_filled_accessor.h"
#include "kernels/emulated_parallel_tensor/shard_signature_instance_is_valid.h"
#include "kernels/local_cpu_allocator.h"
#include "kernels/transpose_kernels_cpu.h"
#include "op-attrs/computation_graph_op_attrs.dtg.h"
#include "op-attrs/parallel_tensor_shape.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

static std::pair<ff_dim_t, ff_dim_t> map_dim(int src, int dst) {
  return std::pair{
      ff_dim_t{nonnegative_int{dst}},
      ff_dim_t{nonnegative_int{src}},
  };
}

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("transpose_get_output_shape") {
    TensorShape input_shape = TensorShape{
        TensorDims{
            FFOrdered{
                14_p,
                8_p,
                12_p,
                16_p,
            },
        },
        DataType::FLOAT,
    };

    TransposeAttrs attrs = TransposeAttrs{
        TensorDimPermutation{
            bidict<ff_dim_t, ff_dim_t>{
                map_dim(0, 1),
                map_dim(2, 0),
                map_dim(1, 2),
                map_dim(3, 3),
            },
        },
    };

    TensorShape result = transpose_get_output_shape(attrs, input_shape);
    TensorShape correct = TensorShape{
        TensorDims{
            FFOrdered{
                12_p,
                14_p,
                8_p,
                16_p,
            },
        },
        DataType::FLOAT,
    };

    CHECK(result == correct);
  }

  TEST_CASE("transpose_get_output_parallel_dim_degrees") {
    ParallelTensorDimDegrees input_dim_degrees = ParallelTensorDimDegrees{
        /*sum_degree=*/SumDegree{2_p},
        /*discard_copy_degree=*/DiscardCopyDegree{3_p},
        /*shard_degrees=*/
        FFOrdered{
            2_p,
            5_p,
            3_p,
            7_p,
        },
    };

    TransposeAttrs attrs = TransposeAttrs{
        TensorDimPermutation{
            bidict<ff_dim_t, ff_dim_t>{
                map_dim(0, 1),
                map_dim(2, 0),
                map_dim(1, 2),
                map_dim(3, 3),
            },
        },
    };

    ParallelTensorDimDegrees result =
        transpose_get_output_parallel_dim_degrees(attrs, input_dim_degrees);
    ParallelTensorDimDegrees correct = ParallelTensorDimDegrees{
        /*sum_degree=*/SumDegree{2_p},
        /*discard_copy_degree=*/DiscardCopyDegree{3_p},
        /*shard_degrees=*/
        FFOrdered{
            3_p,
            2_p,
            5_p,
            7_p,
        },
    };

    CHECK(result == correct);
  }

  TEST_CASE("transpose_get_output_parallel_shape") {
    TransposeAttrs attrs = TransposeAttrs{
        TensorDimPermutation{
            bidict<ff_dim_t, ff_dim_t>{
                map_dim(0, 3),
                map_dim(1, 0),
                map_dim(3, 1),
                map_dim(2, 2),
            },
        },
    };

    ParallelTensorShape input_shape = ParallelTensorShape{
        ParallelTensorDims{
            /*shard_dims=*/FFOrdered{
                ShardParallelDim{14_p, 7_p},
                ShardParallelDim{8_p, 2_p},
                ShardParallelDim{12_p, 4_p},
                ShardParallelDim{16_p, 8_p},
            },
            /*replica_dims=*/
            ReplicaParallelDimSet{
                SumDegree{5_p},
                DiscardCopyDegree{3_p},
            },
        },
        DataType::FLOAT,
    };

    ParallelTensorShape result =
        transpose_get_output_parallel_shape(attrs, input_shape);
    ParallelTensorShape correct = ParallelTensorShape{
        ParallelTensorDims{
            /*shard_dims=*/FFOrdered{
                ShardParallelDim{8_p, 2_p},
                ShardParallelDim{16_p, 8_p},
                ShardParallelDim{12_p, 4_p},
                ShardParallelDim{14_p, 7_p},
            },
            /*replica_dims=*/
            ReplicaParallelDimSet{
                SumDegree{5_p},
                DiscardCopyDegree{3_p},
            },
        },
        DataType::FLOAT,
    };

    CHECK(result == correct);
  }

  TEST_CASE("transpose_get_task_group") {
    Allocator cpu_allocator = create_local_cpu_memory_allocator();

    TransposeAttrs attrs = TransposeAttrs{
        TensorDimPermutation{
            bidict<ff_dim_t, ff_dim_t>{
                map_dim(0, 1),
                map_dim(2, 0),
                map_dim(1, 2),
            },
        },
    };

    TensorShape input_shape = TensorShape{
        TensorDims{
            FFOrdered{
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

    auto run_transpose =
        [&](std::map<TensorSlotName, GenericTensorAccessorR> const
                &incoming_shards)
        -> std::map<TensorSlotName, GenericTensorAccessorR> {
      GenericTensorAccessorR input_shard =
          incoming_shards.at(TensorSlotName::INPUT);
      TensorShape output_shard_shape = transpose_get_output_shape(
          attrs, get_tensor_shape_for_accessor_r(input_shard));
      GenericTensorAccessorW output_shard =
          create_zero_filled_accessor_w(output_shard_shape, cpu_allocator);

      transpose_cpu_forward_kernel(
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

    auto transpose_shard_signature_instance_is_valid =
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
          /*run_op=*/run_transpose,
          /*seed=*/0);
    };

    SUBCASE("data parallelism") {
      ParallelTensorDimDegrees input_dim_degrees =
          mk_dim_degrees(1, 1, 2, 1, 1);

      CHECK(transpose_shard_signature_instance_is_valid(input_dim_degrees));
    }

    SUBCASE("mixed parallelism") {
      ParallelTensorDimDegrees input_dim_degrees =
          mk_dim_degrees(3, 2, 3, 2, 1);

      CHECK(transpose_shard_signature_instance_is_valid(input_dim_degrees));
    }
  }
}
