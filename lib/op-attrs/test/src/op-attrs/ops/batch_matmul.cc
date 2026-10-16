#include "op-attrs/ops/batch_matmul.h"
#include "kernels/batch_matmul_kernels_cpu.h"
#include "kernels/create_zero_filled_accessor.h"
#include "kernels/emulated_parallel_tensor/emulated_parallel_tensor.h"
#include "kernels/emulated_parallel_tensor/parallel_tensor_reduction.h"
#include "kernels/emulated_parallel_tensor/shard_signature_instance_is_valid.h"
#include "kernels/format_accessor_contents.h"
#include "kernels/local_cpu_allocator.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "test/utils/doctest/check_kv.h"
#include "utils/containers/require_only_key.h"
#include "utils/containers/set_filtrans.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("batch_matmul_get_output_shape") {
    BatchMatmulAttrs attrs = BatchMatmulAttrs{};

    TensorShape lhs = TensorShape{
        TensorDims{
            FFOrdered<positive_int>{
                5_p,
                4_p,
                12_p,
                3_p,
            },
        },
        DataType::FLOAT,
    };

    SUBCASE("inner dimensions match") {
      TensorShape rhs = TensorShape{
          TensorDims{
              FFOrdered<positive_int>{
                  5_p,
                  4_p,
                  3_p,
                  8_p,
              },
          },
          DataType::FLOAT,
      };

      TensorShape result = batch_matmul_get_output_shape(attrs, lhs, rhs);

      TensorShape correct = TensorShape{
          TensorDims{
              FFOrdered<positive_int>{
                  5_p,
                  4_p,
                  12_p,
                  8_p,
              },
          },
          DataType::FLOAT,
      };

      CHECK(result == correct);
    }

    SUBCASE("inner dimensions don't match") {
      TensorShape rhs = TensorShape{
          TensorDims{
              FFOrdered<positive_int>{
                  5_p,
                  4_p,
                  4_p,
                  8_p,
              },
          },
          DataType::FLOAT,
      };

      CHECK_THROWS(batch_matmul_get_output_shape(attrs, lhs, rhs));
    }

    SUBCASE("leading dimensions don't match") {
      TensorShape rhs = TensorShape{
          TensorDims{
              FFOrdered<positive_int>{
                  5_p,
                  6_p,
                  3_p,
                  8_p,
              },
          },
          DataType::FLOAT,
      };

      CHECK_THROWS(batch_matmul_get_output_shape(attrs, lhs, rhs));
    }
  }

  TEST_CASE("batch_matmul_get_output_parallel_dim_degrees") {
    BatchMatmulAttrs attrs = BatchMatmulAttrs{};

    auto mk_degrees = [](positive_int sum_degree,
                         positive_int discard_copy_degree,
                         positive_int batch_dim_degree,
                         positive_int row_degree,
                         positive_int col_degree) -> ParallelTensorDimDegrees {
      return ParallelTensorDimDegrees{
          /*sum_degree=*/SumDegree{sum_degree},
          /*discard_copy_degree=*/DiscardCopyDegree{discard_copy_degree},
          /*shard_degrees=*/
          FFOrdered<positive_int>{
              batch_dim_degree,
              row_degree,
              col_degree,
          },
      };
    };

    SUBCASE("data parallelism") {
      SUBCASE("degrees match") {
        ParallelTensorDimDegrees result =
            batch_matmul_get_output_parallel_dim_degrees(
                /*attrs=*/attrs,
                /*lhs=*/mk_degrees(1_p, 1_p, 3_p, 1_p, 1_p),
                /*rhs=*/mk_degrees(1_p, 1_p, 3_p, 1_p, 1_p));

        ParallelTensorDimDegrees correct = mk_degrees(1_p, 1_p, 3_p, 1_p, 1_p);

        CHECK(result == correct);
      }

      SUBCASE("degrees don't match") {
        CHECK_THROWS(batch_matmul_get_output_parallel_dim_degrees(
            /*attrs=*/attrs,
            /*lhs=*/mk_degrees(1_p, 1_p, 3_p, 1_p, 1_p),
            /*rhs=*/mk_degrees(1_p, 1_p, 4_p, 1_p, 1_p)));
      }
    }

    SUBCASE("reduction parallelism") {
      SUBCASE("degrees match") {
        ParallelTensorDimDegrees result =
            batch_matmul_get_output_parallel_dim_degrees(
                /*attrs=*/attrs,
                /*lhs=*/mk_degrees(1_p, 1_p, 1_p, 1_p, 5_p),
                /*rhs=*/mk_degrees(1_p, 1_p, 1_p, 5_p, 1_p));

        ParallelTensorDimDegrees correct = mk_degrees(5_p, 1_p, 1_p, 1_p, 1_p);

        CHECK(result == correct);
      }

      SUBCASE("degrees don't match") {
        CHECK_THROWS(batch_matmul_get_output_parallel_dim_degrees(
            /*attrs=*/attrs,
            /*lhs=*/mk_degrees(1_p, 1_p, 1_p, 1_p, 5_p),
            /*rhs=*/mk_degrees(1_p, 1_p, 1_p, 4_p, 1_p)));
      }
    }

    SUBCASE("lhs row parallelism") {
      SUBCASE("degrees match") {
        ParallelTensorDimDegrees result =
            batch_matmul_get_output_parallel_dim_degrees(
                /*attrs=*/attrs,
                /*lhs=*/mk_degrees(1_p, 1_p, 1_p, 2_p, 1_p),
                /*rhs=*/mk_degrees(1_p, 2_p, 1_p, 1_p, 1_p));

        ParallelTensorDimDegrees correct = mk_degrees(1_p, 1_p, 1_p, 2_p, 1_p);

        CHECK(result == correct);
      }

      SUBCASE("degrees don't match") {
        CHECK_THROWS(batch_matmul_get_output_parallel_dim_degrees(
            /*attrs=*/attrs,
            /*lhs=*/mk_degrees(1_p, 1_p, 1_p, 2_p, 1_p),
            /*rhs=*/mk_degrees(1_p, 3_p, 1_p, 1_p, 1_p)));
      }
    }

    SUBCASE("rhs column parallelism") {
      SUBCASE("degrees match") {
        ParallelTensorDimDegrees result =
            batch_matmul_get_output_parallel_dim_degrees(
                /*attrs=*/attrs,
                /*lhs=*/mk_degrees(1_p, 3_p, 1_p, 1_p, 1_p),
                /*rhs=*/mk_degrees(1_p, 1_p, 1_p, 1_p, 3_p));

        ParallelTensorDimDegrees correct = mk_degrees(1_p, 1_p, 1_p, 1_p, 3_p);

        CHECK(result == correct);
      }

      SUBCASE("degrees don't match") {
        CHECK_THROWS(batch_matmul_get_output_parallel_dim_degrees(
            /*attrs=*/attrs,
            /*lhs=*/mk_degrees(1_p, 4_p, 1_p, 1_p, 1_p),
            /*rhs=*/mk_degrees(1_p, 1_p, 1_p, 1_p, 2_p)));
      }
    }

    SUBCASE("pre-existing lhs sum parallelism") {
      SUBCASE("degrees match") {
        ParallelTensorDimDegrees result =
            batch_matmul_get_output_parallel_dim_degrees(
                /*attrs=*/attrs,
                /*lhs=*/mk_degrees(3_p, 1_p, 1_p, 1_p, 1_p),
                /*rhs=*/mk_degrees(1_p, 3_p, 1_p, 1_p, 1_p));

        ParallelTensorDimDegrees correct = mk_degrees(3_p, 1_p, 1_p, 1_p, 1_p);

        CHECK(result == correct);
      }

      SUBCASE("degrees don't match") {
        CHECK_THROWS(batch_matmul_get_output_parallel_dim_degrees(
            /*attrs=*/attrs,
            /*lhs=*/mk_degrees(3_p, 1_p, 1_p, 1_p, 1_p),
            /*rhs=*/mk_degrees(1_p, 2_p, 1_p, 1_p, 1_p)));
      }
    }

    SUBCASE("pre-existing rhs sum parallelism") {
      SUBCASE("degrees match") {
        ParallelTensorDimDegrees result =
            batch_matmul_get_output_parallel_dim_degrees(
                /*attrs=*/attrs,
                /*lhs=*/mk_degrees(1_p, 5_p, 1_p, 1_p, 1_p),
                /*rhs=*/mk_degrees(5_p, 1_p, 1_p, 1_p, 1_p));

        ParallelTensorDimDegrees correct = mk_degrees(5_p, 1_p, 1_p, 1_p, 1_p);

        CHECK(result == correct);
      }

      SUBCASE("degrees don't match") {
        CHECK_THROWS(batch_matmul_get_output_parallel_dim_degrees(
            /*attrs=*/attrs,
            /*lhs=*/mk_degrees(1_p, 4_p, 1_p, 1_p, 1_p),
            /*rhs=*/mk_degrees(5_p, 1_p, 1_p, 1_p, 1_p)));
      }
    }

    SUBCASE("pre-existing lhs and rhs sum parallelism") {
      SUBCASE("degrees match") {
        ParallelTensorDimDegrees result =
            batch_matmul_get_output_parallel_dim_degrees(
                /*attrs=*/attrs,
                /*lhs=*/mk_degrees(3_p, 2_p, 1_p, 1_p, 1_p),
                /*rhs=*/mk_degrees(2_p, 3_p, 1_p, 1_p, 1_p));

        ParallelTensorDimDegrees correct = mk_degrees(6_p, 1_p, 1_p, 1_p, 1_p);

        CHECK(result == correct);
      }

      SUBCASE("degrees don't match") {
        CHECK_THROWS(batch_matmul_get_output_parallel_dim_degrees(
            /*attrs=*/attrs,
            /*lhs=*/mk_degrees(2_p, 3_p, 1_p, 1_p, 1_p),
            /*rhs=*/mk_degrees(2_p, 3_p, 1_p, 1_p, 1_p)));
      }
    }

    SUBCASE("all the degrees at once") {
      ParallelTensorDimDegrees result =
          batch_matmul_get_output_parallel_dim_degrees(
              /*attrs=*/attrs,
              /*lhs=*/mk_degrees(13_p, 11_p * 3_p, 2_p, 7_p, 5_p),
              /*rhs=*/mk_degrees(11_p, 13_p * 7_p, 2_p, 5_p, 3_p));

      ParallelTensorDimDegrees correct =
          mk_degrees(13_p * 11_p * 5_p, 1_p, 2_p, 7_p, 3_p);

      CHECK(result == correct);
    }
  }

  TEST_CASE("batch_matmul_get_output_parallel_shape") {
    // since most of the edge cases are already tested in
    // batch_matmul_get_output_shape and batch_matmul_get_output_parallel_dim_degrees,
    // here we just do a basic check that they compose

    BatchMatmulAttrs attrs = BatchMatmulAttrs{};

    ParallelTensorShape lhs_shape = ParallelTensorShape{
        ParallelTensorDims{
            FFOrdered<ShardParallelDim>{
                ShardParallelDim{4_p, 2_p},
                ShardParallelDim{14_p, 7_p},
                ShardParallelDim{10_p, 5_p},
            },
            ReplicaParallelDimSet{
                SumDegree{13_p},
                DiscardCopyDegree{11_p * 3_p},
            },
        },
        DataType::FLOAT,
    };

    ParallelTensorShape rhs_shape = ParallelTensorShape{
        ParallelTensorDims{
            FFOrdered<ShardParallelDim>{
                ShardParallelDim{4_p, 2_p},
                ShardParallelDim{10_p, 5_p},
                ShardParallelDim{6_p, 3_p},
            },
            ReplicaParallelDimSet{
                SumDegree{11_p},
                DiscardCopyDegree{13_p * 7_p},
            },
        },
        DataType::FLOAT,
    };

    ParallelTensorShape result =
        batch_matmul_get_output_parallel_shape(attrs, lhs_shape, rhs_shape);

    ParallelTensorShape correct = ParallelTensorShape{
        ParallelTensorDims{
            FFOrdered<ShardParallelDim>{
                ShardParallelDim{4_p, 2_p},
                ShardParallelDim{14_p, 7_p},
                ShardParallelDim{6_p, 3_p},
            },
            ReplicaParallelDimSet{
                SumDegree{13_p * 11_p * 5_p},
                DiscardCopyDegree{1_p},
            },
        },
        DataType::FLOAT,
    };

    CHECK(result == correct);
  }

  TEST_CASE("batch_matmul_get_task_group") {
    Allocator cpu_allocator = create_local_cpu_memory_allocator();

    auto mk_degrees = [](int sum_degree,
                         int discard_copy_degree,
                         int batch_dim_degree,
                         int row_degree,
                         int col_degree) -> ParallelTensorDimDegrees {
      return ParallelTensorDimDegrees{
          /*sum_degree=*/SumDegree{positive_int{sum_degree}},
          /*discard_copy_degree=*/
          DiscardCopyDegree{positive_int{discard_copy_degree}},
          /*shard_degrees=*/
          FFOrdered<positive_int>{
              positive_int{batch_dim_degree},
              positive_int{row_degree},
              positive_int{col_degree},
          },
      };
    };

    TensorShape lhs_shard_shape = TensorShape{
        /*dims=*/TensorDims{
            FFOrdered<positive_int>{
                4_p,
                2_p,
                3_p,
            },
        },
        /*data_type=*/DataType::FLOAT,
    };

    TensorShape rhs_shard_shape = TensorShape{
        /*dims=*/TensorDims{
            FFOrdered<positive_int>{
                4_p,
                3_p,
                1_p,
            },
        },
        /*data_type=*/DataType::FLOAT,
    };

    BatchMatmulAttrs attrs = BatchMatmulAttrs{};

    auto run_bmm = [&](std::map<TensorSlotName, GenericTensorAccessorR> const
                           &input_shards)
        -> std::map<TensorSlotName, GenericTensorAccessorR> {
      GenericTensorAccessorR lhs_input_shard =
          input_shards.at(TensorSlotName::LHS_INPUT);
      GenericTensorAccessorR rhs_input_shard =
          input_shards.at(TensorSlotName::RHS_INPUT);
      TensorShape output_shard_shape = batch_matmul_get_output_shape(
          attrs, lhs_input_shard.shape, rhs_input_shard.shape);
      GenericTensorAccessorW output_shard =
          create_zero_filled_accessor_w(output_shard_shape, cpu_allocator);

      batch_matmul_cpu_forward_kernel(
          /*input_lhs=*/lhs_input_shard,
          /*input_rhs=*/rhs_input_shard,
          /*output=*/output_shard);

      return std::map<TensorSlotName, GenericTensorAccessorR>{
          {
              TensorSlotName::OUTPUT,
              read_only_accessor_from_write_accessor(output_shard),
          },
      };
    };

    auto bmm_shard_signature_instance_is_valid =
        [&](ParallelTensorDimDegrees const &lhs_degrees,
            ParallelTensorDimDegrees const &rhs_degrees) -> bool {
      ParallelTensorShape lhs_shape =
          lift_shape_to_parallel_with_degrees(lhs_shard_shape, lhs_degrees);
      ParallelTensorShape rhs_shape =
          lift_shape_to_parallel_with_degrees(rhs_shard_shape, rhs_degrees);

      std::map<TensorSlotName, ParallelTensorShape> input_shapes = {
          {
              TensorSlotName::LHS_INPUT,
              lhs_shape,
          },
          {
              TensorSlotName::RHS_INPUT,
              rhs_shape,
          },
      };

      return shard_signature_instance_is_valid(
          /*attrs=*/ComputationGraphOpAttrs{BatchMatmulAttrs{}},
          /*input_shapes=*/input_shapes,
          /*run_op=*/run_bmm,
          /*seed=*/0);
    };

    SUBCASE("data parallelism") {
      ParallelTensorDimDegrees lhs_degrees = mk_degrees(1, 1, 2, 1, 1);
      ParallelTensorDimDegrees rhs_degrees = mk_degrees(1, 1, 2, 1, 1);

      CHECK(bmm_shard_signature_instance_is_valid(lhs_degrees, rhs_degrees));
    }

    SUBCASE("reduction parallelism") {
      ParallelTensorDimDegrees lhs_degrees = mk_degrees(1, 1, 1, 1, 3);
      ParallelTensorDimDegrees rhs_degrees = mk_degrees(1, 1, 1, 3, 1);

      CHECK(bmm_shard_signature_instance_is_valid(lhs_degrees, rhs_degrees));
    }
  }
}
