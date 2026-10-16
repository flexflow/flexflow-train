#include "op-attrs/ops/pool_2d.h"
#include "kernels/accessor.h"
#include "kernels/create_zero_filled_accessor.h"
#include "kernels/emulated_parallel_tensor/shard_signature_instance_is_valid.h"
#include "kernels/local_cpu_allocator.h"
#include "kernels/pool_2d_kernels_cpu.h"
#include "op-attrs/parallel_tensor_dims.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "utils/fmt/optional.h"
#include "utils/integer_conversions.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("make_adaptive_pool2d") {
    positive_int input_n = 10_p;
    positive_int input_c = 11_p;
    positive_int input_h = 15_p;
    positive_int input_w = 20_p;
    Activation activation = Activation::RELU;
    PoolOp op = PoolOp::AVG;

    TensorDims input_dims =
        TensorDims{FFOrdered{input_n, input_c, input_h, input_w}};

    SUBCASE("input_h divisible by output_h && input_w divisible by output_w") {
      positive_int output_h = 5_p;
      positive_int output_w = 2_p;

      Pool2DAttrs correct_attrs = Pool2DAttrs{
          /*kernel_h=*/3_p,
          /*kernel_w=*/10_p,
          /*stride_h=*/3_p,
          /*stride_w=*/10_p,
          /*padding_h=*/0_n,
          /*padding_w=*/0_n,
          /*pool_type=*/op,
          /*activation=*/activation,
      };

      SUBCASE("returns correct attrs") {
        Pool2DAttrs result = make_adaptive_pool2d_attrs(
            input_dims, output_h, output_w, op, activation);
        Pool2DAttrs correct = correct_attrs;

        CHECK(result == correct);
      }

      SUBCASE(
          "confirm that output shape is as expected for the expected attrs") {
        TensorShape input_shape = TensorShape{input_dims, DataType::FLOAT};

        TensorShape result =
            pool2d_get_output_shape(correct_attrs, input_shape);
        TensorShape correct = TensorShape{
            TensorDims{FFOrdered{
                input_n,
                input_c,
                output_h,
                output_w,
            }},
            DataType::FLOAT,
        };

        CHECK(result == correct);
      }
    }

    SUBCASE("input_h not divisible by output_h") {
      positive_int output_h = 6_p;
      positive_int output_w = 2_p;

      CHECK_THROWS(make_adaptive_pool2d_attrs(
          input_dims, output_h, output_w, op, activation));
    }

    SUBCASE("input_w not divisible by output_w") {
      positive_int output_h = 5_p;
      positive_int output_w = 3_p;

      CHECK_THROWS(make_adaptive_pool2d_attrs(
          input_dims, output_h, output_w, op, activation));
    }

    SUBCASE("input_h == output_h and input_w == output_w") {
      positive_int output_h = input_h;
      positive_int output_w = input_w;

      Pool2DAttrs correct_attrs = Pool2DAttrs{
          /*kernel_h=*/1_p,
          /*kernel_w=*/1_p,
          /*stride_h=*/1_p,
          /*stride_w=*/1_p,
          /*padding_h=*/0_n,
          /*padding_w=*/0_n,
          /*pool_type=*/op,
          /*activation=*/activation,
      };

      SUBCASE("returns correct attrs") {
        Pool2DAttrs result = make_adaptive_pool2d_attrs(
            input_dims, output_h, output_w, op, activation);
        Pool2DAttrs correct = correct_attrs;

        CHECK(result == correct);
      }

      SUBCASE(
          "confirm that output shape is as expected for the expected attrs") {
        TensorShape input_shape = TensorShape{input_dims, DataType::FLOAT};

        TensorShape result =
            pool2d_get_output_shape(correct_attrs, input_shape);
        TensorShape correct = input_shape;

        CHECK(result == correct);
      }
    }
  }

  TEST_CASE("pool2d_get_output_shape") {
    Pool2DAttrs attrs = Pool2DAttrs{
        /*kernel_h=*/3_p,
        /*kernel_w=*/2_p,
        /*stride_h=*/2_p,
        /*stride_w=*/2_p,
        /*padding_h=*/1_n,
        /*padding_w=*/1_n,
        /*pool_type=*/PoolOp::MAX,
        /*activation=*/std::nullopt,
    };

    SUBCASE("fails on 1d inputs") {
      TensorShape input = TensorShape{
          TensorDims{FFOrdered{
              14_p,
          }},
          DataType::FLOAT,
      };

      CHECK_THROWS(pool2d_get_output_shape(attrs, input));
    }

    SUBCASE("2d input") {
      TensorShape input = TensorShape{
          TensorDims{FFOrdered{
              12_p,
              14_p,
          }},
          DataType::FLOAT,
      };

      TensorShape result = pool2d_get_output_shape(attrs, input);
      TensorShape correct = TensorShape{
          TensorDims{FFOrdered{6_p, 8_p}},
          DataType::FLOAT,
      };

      CHECK(result == correct);
    }

    SUBCASE("3d input") {
      TensorShape input = TensorShape{
          TensorDims{FFOrdered{
              10_p,
              12_p,
              14_p,
          }},
          DataType::FLOAT,
      };

      TensorShape result = pool2d_get_output_shape(attrs, input);
      TensorShape correct = TensorShape{
          TensorDims{FFOrdered{10_p, 6_p, 8_p}},
          DataType::FLOAT,
      };

      CHECK(result == correct);
    }

    SUBCASE("4d input") {
      TensorShape input = TensorShape{
          TensorDims{FFOrdered{11_p, 13_p, 12_p, 6_p}},
          DataType::FLOAT,
      };

      TensorShape result = pool2d_get_output_shape(attrs, input);
      TensorShape correct = TensorShape{
          TensorDims{FFOrdered{11_p, 13_p, 6_p, 4_p}},
          DataType::FLOAT,
      };

      CHECK(result == correct);
    }

    SUBCASE("5d input") {
      TensorShape input = TensorShape{
          TensorDims{FFOrdered{11_p, 13_p, 16_p, 12_p, 14_p}},
          DataType::FLOAT,
      };

      TensorShape result = pool2d_get_output_shape(attrs, input);
      TensorShape correct = TensorShape{
          TensorDims{FFOrdered{11_p, 13_p, 16_p, 6_p, 8_p}},
          DataType::FLOAT,
      };

      CHECK(result == correct);
    }
  }

  TEST_CASE("pool2d_get_output_parallel_dim_degrees") {
    auto make_attrs = [](PoolOp pool_type,
                         std::optional<Activation> const &activation) {
      return Pool2DAttrs{
          /*kernel_h=*/3_p,
          /*kernel_w=*/2_p,
          /*stride_h=*/2_p,
          /*stride_w=*/2_p,
          /*padding_h=*/1_n,
          /*padding_w=*/1_n,
          /*pool_type=*/pool_type,
          /*activation=*/activation,
      };
    };

    SUBCASE("allows data parallelism") {
      Pool2DAttrs attrs = make_attrs(PoolOp::MAX, /*activation=*/std::nullopt);

      ParallelTensorDimDegrees input = ParallelTensorDimDegrees{
          SumDegree{1_p},
          DiscardCopyDegree{1_p},
          FFOrdered{
              4_p,
              1_p,
              1_p,
              1_p,
          },
      };

      ParallelTensorDimDegrees result =
          pool2d_get_output_parallel_dim_degrees(attrs, input);
      ParallelTensorDimDegrees correct = input;

      CHECK(result == correct);
    }

    SUBCASE("allows arbitrary input sharding parallelism") {
      Pool2DAttrs attrs = make_attrs(PoolOp::MAX, /*activation=*/std::nullopt);

      ParallelTensorDimDegrees input = ParallelTensorDimDegrees{
          SumDegree{1_p},
          DiscardCopyDegree{1_p},
          FFOrdered{
              4_p,
              2_p,
              5_p,
              6_p,
          },
      };

      ParallelTensorDimDegrees result =
          pool2d_get_output_parallel_dim_degrees(attrs, input);
      ParallelTensorDimDegrees correct = input;

      CHECK(result == correct);
    }

    SUBCASE("allows discard copy parallelism") {
      Pool2DAttrs attrs = make_attrs(PoolOp::MAX, /*activation=*/std::nullopt);

      ParallelTensorDimDegrees input = ParallelTensorDimDegrees{
          SumDegree{1_p},
          DiscardCopyDegree{3_p},
          FFOrdered{
              1_p,
              1_p,
              1_p,
              1_p,
          },
      };

      ParallelTensorDimDegrees result =
          pool2d_get_output_parallel_dim_degrees(attrs, input);
      ParallelTensorDimDegrees correct = input;

      CHECK(result == correct);
    }

    SUBCASE("sum parallelism") {
      SUBCASE("without activation") {
        SUBCASE("PoolOp::MAX does not allow sum parallelism") {
          Pool2DAttrs attrs =
              make_attrs(PoolOp::MAX, /*activation=*/std::nullopt);

          ParallelTensorDimDegrees input = ParallelTensorDimDegrees{
              SumDegree{2_p},
              DiscardCopyDegree{1_p},
              FFOrdered{
                  1_p,
                  1_p,
                  1_p,
                  1_p,
              },
          };

          CHECK_THROWS(pool2d_get_output_parallel_dim_degrees(attrs, input));
        }

        SUBCASE("PoolOp::AVG does allow sum parallelism") {
          Pool2DAttrs attrs =
              make_attrs(PoolOp::AVG, /*activation=*/std::nullopt);

          ParallelTensorDimDegrees input = ParallelTensorDimDegrees{
              SumDegree{2_p},
              DiscardCopyDegree{1_p},
              FFOrdered{
                  1_p,
                  1_p,
                  1_p,
                  1_p,
              },
          };

          ParallelTensorDimDegrees result =
              pool2d_get_output_parallel_dim_degrees(attrs, input);
          ParallelTensorDimDegrees correct = input;

          CHECK(result == correct);
        }
      }

      SUBCASE("with activation does not allow sum parallelism") {
        Pool2DAttrs attrs =
            make_attrs(PoolOp::AVG, /*activation=*/Activation::RELU);

        ParallelTensorDimDegrees input = ParallelTensorDimDegrees{
            SumDegree{2_p},
            DiscardCopyDegree{1_p},
            FFOrdered{
                1_p,
                1_p,
                1_p,
                1_p,
            },
        };

        CHECK_THROWS(pool2d_get_output_parallel_dim_degrees(attrs, input));
      }
    }
  }

  TEST_CASE("pool2d_get_output_parallel_shape") {
    // this function is mostly covered by the tests above, so we
    // just do a single test to make sure it works/exists

    Pool2DAttrs attrs = Pool2DAttrs{
        /*kernel_h=*/3_p,
        /*kernel_w=*/2_p,
        /*stride_h=*/2_p,
        /*stride_w=*/2_p,
        /*padding_h=*/1_n,
        /*padding_w=*/1_n,
        /*pool_type=*/PoolOp::MAX,
        /*activation=*/std::nullopt,
    };

    SUBCASE("valid parallelism") {
      ParallelTensorShape input = ParallelTensorShape{
          ParallelTensorDims{
              FFOrdered<ShardParallelDim>{
                  ShardParallelDim{14_p, 7_p},
                  ShardParallelDim{16_p, 8_p},
                  ShardParallelDim{12_p, 3_p},
                  ShardParallelDim{6_p, 2_p},
              },
              ReplicaParallelDimSet{
                  SumDegree{1_p},
                  DiscardCopyDegree{2_p},
              },
          },
          DataType::FLOAT,
      };

      ParallelTensorShape result =
          pool2d_get_output_parallel_shape(attrs, input);
      ParallelTensorShape correct = ParallelTensorShape{
          ParallelTensorDims{
              FFOrdered<ShardParallelDim>{
                  ShardParallelDim{14_p, 7_p},
                  ShardParallelDim{16_p, 8_p},
                  ShardParallelDim{6_p, 3_p},
                  ShardParallelDim{4_p, 2_p},
              },
              ReplicaParallelDimSet{
                  SumDegree{1_p},
                  DiscardCopyDegree{2_p},
              },
          },
          DataType::FLOAT,
      };
    }

    SUBCASE("invalid parallelism") {
      ParallelTensorShape input = ParallelTensorShape{
          ParallelTensorDims{
              FFOrdered<ShardParallelDim>{
                  ShardParallelDim{14_p, 1_p},
                  ShardParallelDim{16_p, 1_p},
                  ShardParallelDim{12_p, 1_p},
                  ShardParallelDim{6_p, 1_p},
              },
              ReplicaParallelDimSet{
                  SumDegree{2_p},
                  DiscardCopyDegree{1_p},
              },
          },
          DataType::FLOAT,
      };

      CHECK_THROWS(pool2d_get_output_parallel_shape(attrs, input));
    }
  }

  TEST_CASE("pool2d_get_task_group") {
    Allocator cpu_allocator = create_local_cpu_memory_allocator();

    Pool2DAttrs attrs = Pool2DAttrs{
        /*kernel_h=*/3_p,
        /*kernel_w=*/3_p,
        /*stride_h=*/1_p,
        /*stride_w=*/1_p,
        /*padding_h=*/0_n,
        /*padding_w=*/0_n,
        /*pool_type=*/PoolOp::MAX,
        /*activation=*/std::nullopt,
    };

    TensorShape input_shape = TensorShape{
        TensorDims{
            FFOrdered{
                2_p,
                8_p,
                6_p,
                4_p,
            },
        },
        DataType::FLOAT,
    };

    auto mk_dim_degrees =
        [&](int sum_degree,
            int discard_copy_degree,
            int batch_shard_degree,
            int channel_shard_degree,
            int height_shard_degree,
            int width_shard_degree) -> ParallelTensorDimDegrees {
      return ParallelTensorDimDegrees{
          /*sum_degree=*/SumDegree{positive_int{sum_degree}},
          /*discard_copy_degree=*/
          DiscardCopyDegree{positive_int{discard_copy_degree}},
          /*shard_degrees=*/
          FFOrdered{
              positive_int{batch_shard_degree},
              positive_int{channel_shard_degree},
              positive_int{height_shard_degree},
              positive_int{width_shard_degree},
          },
      };
    };

    auto run_pool2d = [&](std::map<TensorSlotName, GenericTensorAccessorR> const
                              &incoming_shards)
        -> std::map<TensorSlotName, GenericTensorAccessorR> {
      GenericTensorAccessorR input_shard =
          incoming_shards.at(TensorSlotName::INPUT);
      TensorShape output_shard_shape = pool2d_get_output_shape(
          attrs, get_tensor_shape_for_accessor_r(input_shard));
      GenericTensorAccessorW output_shard =
          create_zero_filled_accessor_w(output_shard_shape, cpu_allocator);

      pool2d_cpu_forward_kernel(
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

    auto pool2d_shard_signature_instance_is_valid =
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
          /*run_op=*/run_pool2d,
          /*seed=*/0);
    };

    SUBCASE("data parallelism") {
      ParallelTensorDimDegrees input_dim_degrees =
          mk_dim_degrees(1, 1, 2, 1, 1, 1);

      CHECK(pool2d_shard_signature_instance_is_valid(input_dim_degrees));
    }
  }
}
