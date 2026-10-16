#include "op-attrs/ops/reduce.h"
#include "kernels/create_zero_filled_accessor.h"
#include "kernels/emulated_parallel_tensor/shard_signature_instance_is_valid.h"
#include "kernels/local_cpu_allocator.h"
#include "kernels/reduce_kernels_cpu.h"
#include "op-attrs/parallel_tensor_shape.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("reduce_get_output_shape") {
    TensorShape input = TensorShape{
        TensorDims{
            FFOrdered<positive_int>{
                8_p,
                12_p,
                6_p,
                2_p,
            },
        },
        DataType::FLOAT,
    };

    SUBCASE("keepdims = true") {
      ReduceAttrs attrs = ReduceAttrs{
          /*axes=*/nonempty_set<ff_dim_t>{
              ff_dim_t{1_n},
              ff_dim_t{3_n},
          },
          /*op=*/ReduceOp::SUM,
          /*keepdims=*/true,
      };

      TensorShape result = reduce_get_output_shape(attrs, input);
      TensorShape correct = input;

      CHECK(result == correct);
    }

    SUBCASE("keepdims = false") {
      ReduceAttrs attrs = ReduceAttrs{
          /*axes=*/nonempty_set<ff_dim_t>{
              ff_dim_t{1_n},
              ff_dim_t{3_n},
          },
          /*op=*/ReduceOp::SUM,
          /*keepdims=*/false,
      };

      TensorShape result = reduce_get_output_shape(attrs, input);
      TensorShape correct = TensorShape{
          TensorDims{
              FFOrdered<positive_int>{
                  8_p,
                  6_p,
              },
          },
          DataType::FLOAT,
      };

      CHECK(result == correct);
    }
  }

  TEST_CASE("reduce_get_output_parallel_dim_degrees") {
    auto mk_2d = [&](int o_sum,
                     int o_discard,
                     int o_0,
                     int o_1) -> ParallelTensorDimDegrees {
      return ParallelTensorDimDegrees{
          SumDegree{positive_int{o_sum}},
          DiscardCopyDegree{positive_int{o_discard}},
          FFOrdered<positive_int>{
              positive_int{o_0},
              positive_int{o_1},
          },
      };
    };

    auto mk_4d =
        [&](int o_sum, int o_discard, int o_0, int o_1, int o_2, int o_3)
        -> ParallelTensorDimDegrees {
      return ParallelTensorDimDegrees{
          SumDegree{positive_int{o_sum}},
          DiscardCopyDegree{positive_int{o_discard}},
          FFOrdered<positive_int>{
              positive_int{o_0},
              positive_int{o_1},
              positive_int{o_2},
              positive_int{o_3},
          },
      };
    };

    auto mk_attrs = [&](ReduceOp op, bool keepdims = true) -> ReduceAttrs {
      return ReduceAttrs{
          /*axes=*/nonempty_set<ff_dim_t>{
              ff_dim_t{1_n},
              ff_dim_t{3_n},
          },
          /*op=*/op,
          /*keepdims=*/keepdims,
      };
    };

    ReduceAttrs sum_attrs = mk_attrs(ReduceOp::SUM);
    ReduceAttrs max_attrs = mk_attrs(ReduceOp::AVG);

    SUBCASE("pre-existing sum parallelism") {
      ParallelTensorDimDegrees input = mk_4d(3, 1, 1, 1, 1, 1);

      SUBCASE("op = SUM") {
        ParallelTensorDimDegrees result =
            reduce_get_output_parallel_dim_degrees(sum_attrs, input);
        ParallelTensorDimDegrees correct = mk_4d(3, 1, 1, 1, 1, 1);

        CHECK(result == correct);
      }

      SUBCASE("op = AVG") {
        CHECK_THROWS(reduce_get_output_parallel_dim_degrees(max_attrs, input));
      }
    }

    SUBCASE("pre-existing discard copy parallelism") {
      ParallelTensorDimDegrees input = mk_4d(1, 3, 1, 1, 1, 1);

      SUBCASE("op = SUM") {
        ParallelTensorDimDegrees result =
            reduce_get_output_parallel_dim_degrees(sum_attrs, input);
        ParallelTensorDimDegrees correct = mk_4d(1, 3, 1, 1, 1, 1);

        CHECK(result == correct);
      }

      SUBCASE("op = AVG") {
        ParallelTensorDimDegrees result =
            reduce_get_output_parallel_dim_degrees(max_attrs, input);
        ParallelTensorDimDegrees correct = mk_4d(1, 3, 1, 1, 1, 1);

        CHECK(result == correct);
      }
    }

    SUBCASE("shard parallelism not in reduction axes") {
      ParallelTensorDimDegrees input = mk_4d(1, 1, 4, 1, 1, 1);

      SUBCASE("op = SUM") {
        ParallelTensorDimDegrees result =
            reduce_get_output_parallel_dim_degrees(sum_attrs, input);
        ParallelTensorDimDegrees correct = mk_4d(1, 1, 4, 1, 1, 1);

        CHECK(result == correct);
      }

      SUBCASE("op = AVG") {
        ParallelTensorDimDegrees result =
            reduce_get_output_parallel_dim_degrees(max_attrs, input);
        ParallelTensorDimDegrees correct = mk_4d(1, 1, 4, 1, 1, 1);

        CHECK(result == correct);
      }
    }

    SUBCASE("shard parallelism in reduction axis") {
      ParallelTensorDimDegrees input = mk_4d(1, 1, 1, 4, 1, 1);

      SUBCASE("op = SUM") {
        CHECK_THROWS(reduce_get_output_parallel_dim_degrees(sum_attrs, input));
      }

      SUBCASE("op = AVG") {
        CHECK_THROWS(reduce_get_output_parallel_dim_degrees(max_attrs, input));
      }
    }

    SUBCASE("hybrid parallelism") {
      SUBCASE("op = SUM") {
        ParallelTensorDimDegrees input = mk_4d(3, 7, 2, 1, 5, 1);

        ParallelTensorDimDegrees result =
            reduce_get_output_parallel_dim_degrees(sum_attrs, input);
        ParallelTensorDimDegrees correct = mk_4d(3, 7, 2, 1, 5, 1);

        CHECK(result == correct);
      }

      SUBCASE("op = AVG") {
        ParallelTensorDimDegrees input = mk_4d(1, 7, 2, 1, 5, 1);

        ParallelTensorDimDegrees result =
            reduce_get_output_parallel_dim_degrees(max_attrs, input);
        ParallelTensorDimDegrees correct = mk_4d(1, 7, 2, 1, 5, 1);

        CHECK(result == correct);
      }
    }

    SUBCASE("keepdims = false") {
      SUBCASE("op = SUM") {
        ReduceAttrs attrs = mk_attrs(ReduceOp::SUM, /*keepdims=*/false);

        ParallelTensorDimDegrees input = mk_4d(3, 7, 2, 1, 5, 1);

        ParallelTensorDimDegrees result =
            reduce_get_output_parallel_dim_degrees(attrs, input);
        ParallelTensorDimDegrees correct = mk_2d(3, 7, 2, 5);

        CHECK(result == correct);
      }

      SUBCASE("op = AVG") {
        ReduceAttrs attrs = mk_attrs(ReduceOp::AVG, /*keepdims=*/false);

        ParallelTensorDimDegrees input = mk_4d(1, 7, 2, 1, 5, 1);

        ParallelTensorDimDegrees result =
            reduce_get_output_parallel_dim_degrees(attrs, input);
        ParallelTensorDimDegrees correct = mk_2d(1, 7, 2, 5);

        CHECK(result == correct);
      }
    }
  }

  TEST_CASE("reduce_get_output_parallel_shape") {
    // since most of the edge cases are already tested in
    // reduce_get_output_shape and reduce_get_output_parallel_dim_degrees,
    // here we just do a basic check that they compose

    ReduceAttrs attrs = ReduceAttrs{
        /*axes=*/nonempty_set<ff_dim_t>{
            ff_dim_t{1_n},
            ff_dim_t{3_n},
        },
        /*op=*/ReduceOp::SUM,
        /*keepdims=*/false,
    };

    ParallelTensorShape input_shape = ParallelTensorShape{
        ParallelTensorDims{
            FFOrdered<ShardParallelDim>{
                ShardParallelDim{6_p, 2_p},
                ShardParallelDim{7_p, 1_p},
                ShardParallelDim{10_p, 5_p},
                ShardParallelDim{4_p, 1_p},
            },
            ReplicaParallelDimSet{
                SumDegree{7_p},
                DiscardCopyDegree{3_p},
            },
        },
        DataType::FLOAT,
    };

    ParallelTensorShape result =
        reduce_get_output_parallel_shape(attrs, input_shape);

    ParallelTensorShape correct = ParallelTensorShape{
        ParallelTensorDims{
            FFOrdered<ShardParallelDim>{
                ShardParallelDim{6_p, 2_p},
                ShardParallelDim{10_p, 5_p},
            },
            ReplicaParallelDimSet{
                SumDegree{7_p},
                DiscardCopyDegree{3_p},
            },
        },
        DataType::FLOAT,
    };

    CHECK(result == correct);
  }

  TEST_CASE("reduce_get_task_group") {
    Allocator cpu_allocator = create_local_cpu_memory_allocator();

    auto mk_degrees = [](int sum_degree,
                         int discard_copy_degree,
                         int shard_dim_0_degree,
                         int shard_dim_1_degree,
                         int shard_dim_2_degree,
                         int shard_dim_3_degree) -> ParallelTensorDimDegrees {
      return ParallelTensorDimDegrees{
          /*sum_degree=*/SumDegree{positive_int{sum_degree}},
          /*discard_copy_degree=*/
          DiscardCopyDegree{positive_int{discard_copy_degree}},
          /*shard_degrees=*/
          FFOrdered<positive_int>{
              positive_int{shard_dim_0_degree},
              positive_int{shard_dim_1_degree},
              positive_int{shard_dim_2_degree},
              positive_int{shard_dim_3_degree},
          },
      };
    };

    TensorShape input_shard_shape = TensorShape{
        /*dims=*/TensorDims{
            FFOrdered<positive_int>{
                4_p,
                2_p,
                3_p,
                3_p,
            },
        },
        /*data_type=*/DataType::FLOAT,
    };

    auto run_reduce = [&](ReduceAttrs const &attrs,
                          std::map<TensorSlotName, GenericTensorAccessorR> const
                              &input_shards)
        -> std::map<TensorSlotName, GenericTensorAccessorR> {
      GenericTensorAccessorR input_shard =
          input_shards.at(TensorSlotName::INPUT);

      TensorShape output_shard_shape =
          reduce_get_output_shape(attrs, input_shard.shape);
      GenericTensorAccessorW output_shard =
          create_zero_filled_accessor_w(output_shard_shape, cpu_allocator);

      reduce_cpu_forward_kernel(
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

    auto reduce_shard_signature_instance_is_valid =
        [&](ReduceAttrs const &attrs,
            ParallelTensorDimDegrees const &input_degrees) -> bool {
      ParallelTensorShape input_shape =
          lift_shape_to_parallel_with_degrees(input_shard_shape, input_degrees);

      std::map<TensorSlotName, ParallelTensorShape> input_shapes = {
          {
              TensorSlotName::INPUT,
              input_shape,
          },
      };

      auto run_op = [&](std::map<TensorSlotName, GenericTensorAccessorR> const
                            &input_shards)
          -> std::map<TensorSlotName, GenericTensorAccessorR> {
        return run_reduce(attrs, input_shards);
      };

      return shard_signature_instance_is_valid(
          /*attrs=*/ComputationGraphOpAttrs{attrs},
          /*input_shapes=*/input_shapes,
          /*run_op=*/run_op,
          /*seed=*/0);
    };

    auto mk_attrs = [&](ReduceOp op, bool keepdims) -> ReduceAttrs {
      return ReduceAttrs{
          /*axes=*/nonempty_set<ff_dim_t>{
              ff_dim_t{1_n},
              ff_dim_t{3_n},
          },
          /*op=*/op,
          /*keepdims=*/keepdims,
      };
    };

    SUBCASE("op = SUM") {
      SUBCASE("keepdims = true") {
        ReduceAttrs attrs = mk_attrs(
            /*op=*/ReduceOp::SUM,
            /*keepdims=*/true);

        SUBCASE("data parallelism") {
          ParallelTensorDimDegrees input_degrees = mk_degrees(1, 1, 2, 1, 1, 1);

          CHECK(reduce_shard_signature_instance_is_valid(attrs, input_degrees));
        }
      }

      SUBCASE("keepdims = false") {
        ReduceAttrs attrs = mk_attrs(
            /*op=*/ReduceOp::SUM,
            /*keepdims=*/false);

        SUBCASE("data parallelism") {
          ParallelTensorDimDegrees input_degrees = mk_degrees(1, 1, 2, 1, 1, 1);

          CHECK(reduce_shard_signature_instance_is_valid(attrs, input_degrees));
        }
      }
    }

    SUBCASE("op = AVG") {
      SUBCASE("keepdims = true") {
        ReduceAttrs attrs = mk_attrs(
            /*op=*/ReduceOp::AVG,
            /*keepdims=*/true);

        SUBCASE("data parallelism") {
          ParallelTensorDimDegrees input_degrees = mk_degrees(1, 1, 2, 1, 1, 1);

          CHECK(reduce_shard_signature_instance_is_valid(attrs, input_degrees));
        }
      }

      SUBCASE("keepdims = false") {
        ReduceAttrs attrs = mk_attrs(
            /*op=*/ReduceOp::AVG,
            /*keepdims=*/false);

        SUBCASE("data parallelism") {
          ParallelTensorDimDegrees input_degrees = mk_degrees(1, 1, 2, 1, 1, 1);

          CHECK(reduce_shard_signature_instance_is_valid(attrs, input_degrees));
        }
      }
    }
  }
}
