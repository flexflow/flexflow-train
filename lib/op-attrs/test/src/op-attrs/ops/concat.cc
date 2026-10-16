#include "op-attrs/ops/concat.h"
#include "kernels/concat_kernels_cpu.h"
#include "kernels/create_zero_filled_accessor.h"
#include "kernels/emulated_parallel_tensor/shard_signature_instance_is_valid.h"
#include "kernels/local_cpu_allocator.h"
#include "op-attrs/parallel_tensor_dims.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "test/utils/doctest/fmt/expected.h"
#include "test/utils/doctest/fmt/optional.h"
#include "utils/expected.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("concat_get_output_shape") {
    ConcatAttrs attrs = ConcatAttrs{
        /*axis=*/ff_dim_t{1_n},
        /*num_inputs=*/3_ge2,
    };

    SUBCASE("empty input shapes list passed") {
      std::vector<TensorShape> input_shapes = {};

      CHECK_THROWS(concat_get_output_shape(attrs, input_shapes));
    }

    positive_int dim0_size = 12_p;
    positive_int dim2_size = 20_p;
    TensorShape input_shape1 = TensorShape{
        TensorDims{FFOrdered{
            dim0_size,
            14_p,
            dim2_size,
        }},
        DataType::FLOAT,
    };

    SUBCASE("single element input shapes list passed") {
      std::vector<TensorShape> input_shapes = {input_shape1};

      CHECK_THROWS(concat_get_output_shape(attrs, input_shapes));
    }

    TensorShape input_shape2 = TensorShape{
        TensorDims{FFOrdered{
            dim0_size,
            16_p,
            dim2_size,
        }},
        DataType::FLOAT,
    };

    TensorShape input_shape3 = TensorShape{
        TensorDims{FFOrdered{dim0_size, 18_p, dim2_size}},
        DataType::FLOAT,
    };

    SUBCASE("input shapes do not share the same num_dims") {
      TensorShape mismatched_num_dims = TensorShape{
          TensorDims{FFOrdered{
              dim0_size,
              20_p,
              dim2_size,
              1_p,
          }},
          DataType::FLOAT,
      };

      std::vector<TensorShape> input_shapes = {
          input_shape1, input_shape2, input_shape3, mismatched_num_dims};

      CHECK_THROWS(concat_get_output_shape(attrs, input_shapes));
    }

    SUBCASE("concat axis is out of bounds") {
      attrs = ConcatAttrs{
          /*axis=*/ff_dim_t{3_n},
          /*num_inputs=*/3_ge2,
      };

      std::vector<TensorShape> input_shapes = {
          input_shape1, input_shape2, input_shape3};

      CHECK_THROWS(concat_get_output_shape(attrs, input_shapes));
    }

    SUBCASE("input shapes are valid") {
      std::vector<TensorShape> input_shapes = {
          input_shape1, input_shape2, input_shape3};

      TensorShape result = concat_get_output_shape(attrs, input_shapes);
      TensorShape correct = TensorShape{
          TensorDims{FFOrdered{
              dim0_size,
              14_p + 16_p + 18_p,
              dim2_size,
          }},
          DataType::FLOAT,
      };

      CHECK(result == correct);
    }
  }

  TEST_CASE("concat_get_output_parallel_dim_degrees") {
    ConcatAttrs attrs = ConcatAttrs{
        /*axis=*/ff_dim_t{1_n},
        /*num_inputs=*/3_ge2,
    };

    auto mk_3d = [&](int o_sum, int o_eq, int o0, int o1, int o2) {
      return ParallelTensorDimDegrees{
          SumDegree{positive_int{o_sum}},
          DiscardCopyDegree{positive_int{o_eq}},
          FFOrdered<positive_int>{
              positive_int{o0},
              positive_int{o1},
              positive_int{o2},
          },
      };
    };

    SUBCASE("sum reduction parallelism") {
      SUBCASE("matching") {
        SumDegree sum_degree = SumDegree{2_p};

        std::vector<ParallelTensorDimDegrees> inputs = {
            mk_3d(2, 1, 1, 1, 1),
            mk_3d(2, 1, 1, 1, 1),
            mk_3d(2, 1, 1, 1, 1),
        };

        ParallelTensorDimDegrees result =
            concat_get_output_parallel_dim_degrees(attrs, inputs);
        ParallelTensorDimDegrees correct = mk_3d(2, 1, 1, 1, 1);

        CHECK(result == correct);
      }

      SUBCASE("not matching") {
        std::vector<ParallelTensorDimDegrees> inputs = {
            mk_3d(2, 1, 1, 1, 1),
            mk_3d(4, 1, 1, 1, 1),
            mk_3d(4, 1, 1, 1, 1),
        };

        CHECK_THROWS(concat_get_output_parallel_dim_degrees(attrs, inputs));
      }
    }

    SUBCASE("discard copy reduction parallelism") {
      SUBCASE("matching") {
        DiscardCopyDegree discard_copy_degree = DiscardCopyDegree{2_p};

        std::vector<ParallelTensorDimDegrees> inputs = {
            mk_3d(1, 2, 1, 1, 1),
            mk_3d(1, 2, 1, 1, 1),
            mk_3d(1, 2, 1, 1, 1),
        };

        ParallelTensorDimDegrees result =
            concat_get_output_parallel_dim_degrees(attrs, inputs);
        ParallelTensorDimDegrees correct = mk_3d(1, 2, 1, 1, 1);

        CHECK(result == correct);
      }

      SUBCASE("not matching") {
        std::vector<ParallelTensorDimDegrees> inputs = {
            mk_3d(1, 2, 1, 1, 1),
            mk_3d(1, 2, 1, 1, 1),
            mk_3d(1, 4, 1, 1, 1),
        };

        CHECK_THROWS(concat_get_output_parallel_dim_degrees(attrs, inputs));
      }
    }

    SUBCASE("parallelism in axis dim") {
      SUBCASE("matching") {
        positive_int degree = 2_p;

        std::vector<ParallelTensorDimDegrees> inputs = {
            mk_3d(1, 1, 1, 2, 1),
            mk_3d(1, 1, 1, 2, 1),
            mk_3d(1, 1, 1, 2, 1),
        };

        CHECK_THROWS(concat_get_output_parallel_dim_degrees(attrs, inputs));
      }

      SUBCASE("not matching") {
        std::vector<ParallelTensorDimDegrees> inputs = {
            mk_3d(1, 1, 1, 1, 1),
            mk_3d(1, 1, 1, 1, 1),
            mk_3d(1, 1, 1, 2, 1),
        };

        CHECK_THROWS(concat_get_output_parallel_dim_degrees(attrs, inputs));
      }
    }

    SUBCASE("parallelism in non-axis shard dims") {
      SUBCASE("matching") {
        positive_int degree0 = 2_p;
        positive_int degree2 = 4_p;

        std::vector<ParallelTensorDimDegrees> inputs = {
            mk_3d(1, 1, 2, 1, 4),
            mk_3d(1, 1, 2, 1, 4),
            mk_3d(1, 1, 2, 1, 4),
        };

        ParallelTensorDimDegrees result =
            concat_get_output_parallel_dim_degrees(attrs, inputs);
        ParallelTensorDimDegrees correct = mk_3d(1, 1, 2, 1, 4);

        CHECK(result == correct);
      }

      SUBCASE("not matching") {
        std::vector<ParallelTensorDimDegrees> inputs = {
            mk_3d(1, 1, 2, 1, 4),
            mk_3d(1, 1, 4, 1, 2),
            mk_3d(1, 1, 4, 1, 2),
        };

        CHECK_THROWS(concat_get_output_parallel_dim_degrees(attrs, inputs));
      }
    }

    SUBCASE("parallelism degrees are not mutually exclusive") {
      std::vector<ParallelTensorDimDegrees> inputs = {
          mk_3d(3, 5, 2, 1, 4),
          mk_3d(3, 5, 2, 1, 4),
          mk_3d(3, 5, 2, 1, 4),
      };

      ParallelTensorDimDegrees result =
          concat_get_output_parallel_dim_degrees(attrs, inputs);
      ParallelTensorDimDegrees correct = mk_3d(3, 5, 2, 1, 4);

      CHECK(result == correct);
    }
  }

  TEST_CASE("concat_get_output_parallel_shape") {
    // since most of the edge cases are already tested in
    // concat_get_output_shape and concat_get_output_parallel_dim_degrees,
    // here we just do a basic check that they compose

    ConcatAttrs attrs = ConcatAttrs{
        /*axis=*/ff_dim_t{1_n},
        /*num_inputs=*/4_ge2,
    };

    auto mk_input_of_size = [](int x) -> ParallelTensorShape {
      return ParallelTensorShape{
          ParallelTensorDims{
              FFOrdered<ShardParallelDim>{
                  ShardParallelDim{12_p, 2_p},
                  ShardParallelDim{positive_int{x}, 1_p},
                  ShardParallelDim{20_p, 4_p},
              },
              ReplicaParallelDimSet{
                  SumDegree{3_p},
                  DiscardCopyDegree{5_p},
              },
          },
          DataType::INT32,
      };
    };

    std::vector<ParallelTensorShape> inputs = {
        mk_input_of_size(14),
        mk_input_of_size(16),
        mk_input_of_size(14),
        mk_input_of_size(21),
    };

    ParallelTensorShape result =
        concat_get_output_parallel_shape(attrs, inputs);

    ParallelTensorShape correct = mk_input_of_size(14 + 16 + 14 + 21);

    CHECK(result == correct);
  }

  TEST_CASE("concat_get_task_group") {
    Allocator cpu_allocator = create_local_cpu_memory_allocator();

    ConcatAttrs attrs = ConcatAttrs{
        /*axis=*/ff_dim_t{1_n},
        /*num_inputs=*/3_ge2,
    };

    auto mk_tensor_shape =
        [](int dim0_size, int dim1_size, int dim2_size) -> TensorShape {
      return TensorShape{
          TensorDims{
              FFOrdered{
                  positive_int{dim0_size},
                  positive_int{dim1_size},
                  positive_int{dim2_size},
              },
          },
          DataType::FLOAT,
      };
    };

    TensorShape input1_shape = mk_tensor_shape(4, 8, 6);
    TensorShape input2_shape = mk_tensor_shape(4, 4, 6);
    TensorShape input3_shape = mk_tensor_shape(4, 4, 6);

    auto mk_dim_degrees =
        [](int sum_degree,
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

    auto run_concat = [&](std::map<TensorSlotName, GenericTensorAccessorR> const
                              &incoming_shards)
        -> std::map<TensorSlotName, GenericTensorAccessorR> {
      std::vector<GenericTensorAccessorR> input_shards =
          transform(concat_get_input_slot_names(attrs),
                    [&](TensorSlotName slot_name) -> GenericTensorAccessorR {
                      return incoming_shards.at(slot_name);
                    });

      std::vector<TensorShape> input_shard_shapes = transform(
          input_shards,
          [&](GenericTensorAccessorR const &input_shard) -> TensorShape {
            return input_shard.shape;
          });

      TensorShape output_shard_shape =
          concat_get_output_shape(attrs, input_shard_shapes);

      GenericTensorAccessorW output_shard =
          create_zero_filled_accessor_w(output_shard_shape, cpu_allocator);

      concat_cpu_forward_kernel(
          /*output=*/output_shard,
          /*inputs=*/input_shards,
          /*axis=*/attrs.axis);

      return std::map<TensorSlotName, GenericTensorAccessorR>{
          {
              TensorSlotName::OUTPUT,
              read_only_accessor_from_write_accessor(output_shard),
          },
      };
    };

    auto concat_shard_signature_instance_is_valid =
        [&](ParallelTensorDimDegrees const &input1_degrees,
            ParallelTensorDimDegrees const &input2_degrees,
            ParallelTensorDimDegrees const &input3_degrees) -> bool {
      ParallelTensorShape input1_parallel_shape =
          lift_shape_to_parallel_with_degrees(input1_shape, input1_degrees);
      ParallelTensorShape input2_parallel_shape =
          lift_shape_to_parallel_with_degrees(input2_shape, input2_degrees);
      ParallelTensorShape input3_parallel_shape =
          lift_shape_to_parallel_with_degrees(input3_shape, input3_degrees);

      std::map<TensorSlotName, ParallelTensorShape> input_shapes = {
          {
              TensorSlotName::INPUT_00,
              input1_parallel_shape,
          },
          {
              TensorSlotName::INPUT_01,
              input2_parallel_shape,
          },
          {
              TensorSlotName::INPUT_02,
              input3_parallel_shape,
          },
      };

      return shard_signature_instance_is_valid(
          /*attrs=*/ComputationGraphOpAttrs{attrs},
          /*input_shapes=*/input_shapes,
          /*run_op=*/run_concat,
          /*seed=*/0);
    };

    SUBCASE("data parallelism") {
      ParallelTensorDimDegrees input1_dim_degrees =
          mk_dim_degrees(1, 1, 2, 1, 1);
      ParallelTensorDimDegrees input2_dim_degrees = input1_dim_degrees;
      ParallelTensorDimDegrees input3_dim_degrees = input1_dim_degrees;

      CHECK(concat_shard_signature_instance_is_valid(
          input1_dim_degrees, input2_dim_degrees, input3_dim_degrees));
    }

    SUBCASE("inner dimension parallelism") {
      ParallelTensorDimDegrees input1_dim_degrees =
          mk_dim_degrees(1, 1, 1, 1, 2);
      ParallelTensorDimDegrees input2_dim_degrees = input1_dim_degrees;
      ParallelTensorDimDegrees input3_dim_degrees = input1_dim_degrees;

      CHECK(concat_shard_signature_instance_is_valid(
          input1_dim_degrees, input2_dim_degrees, input3_dim_degrees));
    }

    SUBCASE("discard copy parallelism") {
      ParallelTensorDimDegrees input1_dim_degrees =
          mk_dim_degrees(1, 2, 1, 1, 1);
      ParallelTensorDimDegrees input2_dim_degrees = input1_dim_degrees;
      ParallelTensorDimDegrees input3_dim_degrees = input1_dim_degrees;

      CHECK(concat_shard_signature_instance_is_valid(
          input1_dim_degrees, input2_dim_degrees, input3_dim_degrees));
    }

    SUBCASE("sum parallelism") {
      ParallelTensorDimDegrees input1_dim_degrees =
          mk_dim_degrees(2, 1, 1, 1, 1);
      ParallelTensorDimDegrees input2_dim_degrees = input1_dim_degrees;
      ParallelTensorDimDegrees input3_dim_degrees = input1_dim_degrees;

      CHECK(concat_shard_signature_instance_is_valid(
          input1_dim_degrees, input2_dim_degrees, input3_dim_degrees));
    }
  }
}
