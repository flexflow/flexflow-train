#include "op-attrs/ops/element_unary.h"
#include "kernels/accessor.h"
#include "kernels/create_zero_filled_accessor.h"
#include "kernels/element_unary_kernels_cpu.h"
#include "kernels/emulated_parallel_tensor/shard_signature_instance_is_valid.h"
#include "kernels/local_cpu_allocator.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "test/utils/doctest/fmt/expected.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("ReLU shape inference") {
    positive_int d1 = 16_p;
    positive_int d2 = 32_p;
    positive_int d3 = 24_p;

    ElementUnaryAttrs attrs =
        ElementUnaryAttrs{ElementUnaryOp::RELU, std::nullopt};

    TensorShape input = TensorShape{
        TensorDims{
            FFOrdered{
                d1,
                d2,
                d3,
            },
        },
        DataType::FLOAT,
    };

    TensorShape result = element_unary_get_output_shape(attrs, input);
    TensorShape correct = input;

    CHECK(result == correct);

    auto make_input = [&](SumDegree o_sum,
                          DiscardCopyDegree o_eq,
                          positive_int o_1,
                          positive_int o_2,
                          positive_int o_3) -> ParallelTensorShape {
      return lift_shape_to_parallel_with_degrees(
          input, o_sum, o_eq, FFOrdered{o_1, o_2, o_3});
    };

    SUBCASE("partition i.e., sharding parallelism") {
      positive_int degree1 = 4_p;
      positive_int degree2 = 8_p;
      ParallelTensorShape par_input = make_input(
          SumDegree{1_p}, DiscardCopyDegree{1_p}, degree1, 1_p, degree2);

      ParallelTensorShape result =
          element_unary_get_output_parallel_shape(attrs, par_input);
      ParallelTensorShape correct = par_input;

      CHECK(result == correct);
    }

    SUBCASE("discard copy degree > 1") {
      positive_int degree = 2_p;

      ParallelTensorShape par_input =
          make_input(SumDegree{1_p}, DiscardCopyDegree{degree}, 1_p, 1_p, 1_p);

      ParallelTensorShape result =
          element_unary_get_output_parallel_shape(attrs, par_input);
      ParallelTensorShape correct = par_input;

      CHECK(result == correct);
    }

    SUBCASE("sum degree > 1") {
      positive_int degree = 2_p;

      CHECK_THROWS(element_unary_get_output_parallel_shape(
          attrs,
          make_input(
              SumDegree{degree}, DiscardCopyDegree{1_p}, 1_p, 1_p, 1_p)));
    }
  }

  TEST_CASE("element_unary_get_task_group") {
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
                8_p,
            },
        },
        /*data_type=*/DataType::FLOAT,
    };

    auto run_element_unary =
        [&](ElementUnaryAttrs const &attrs,
            std::map<TensorSlotName, GenericTensorAccessorR> const
                &input_shards)
        -> std::map<TensorSlotName, GenericTensorAccessorR> {
      GenericTensorAccessorR input_shard =
          input_shards.at(TensorSlotName::INPUT);

      TensorShape output_shard_shape =
          element_unary_get_output_shape(attrs, input_shard.shape);
      GenericTensorAccessorW output_shard =
          create_zero_filled_accessor_w(output_shard_shape, cpu_allocator);

      element_unary_cpu_forward_kernel(
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

    auto element_unary_shard_signature_instance_is_valid =
        [&](ElementUnaryAttrs const &attrs,
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
        return run_element_unary(attrs, input_shards);
      };

      return shard_signature_instance_is_valid(
          /*attrs=*/ComputationGraphOpAttrs{attrs},
          /*input_shapes=*/input_shapes,
          /*run_op=*/run_op,
          /*seed=*/0);
    };

    SUBCASE("SILU") {
      ElementUnaryAttrs attrs = ElementUnaryAttrs{
          /*op_type=*/ElementUnaryOp::SILU,
          /*scalar=*/std::nullopt,
      };

      SUBCASE("data parallelism") {
        ParallelTensorDimDegrees input_degrees = mk_degrees(1, 1, 2, 1, 1);

        CHECK(element_unary_shard_signature_instance_is_valid(attrs,
                                                              input_degrees));
      }

      SUBCASE("inner dimension 1 parallelism") {
        ParallelTensorDimDegrees input_degrees = mk_degrees(1, 1, 1, 2, 1);

        CHECK(element_unary_shard_signature_instance_is_valid(attrs,
                                                              input_degrees));
      }

      SUBCASE("inner dimension 2 parallelism") {
        ParallelTensorDimDegrees input_degrees = mk_degrees(1, 1, 1, 1, 2);

        CHECK(element_unary_shard_signature_instance_is_valid(attrs,
                                                              input_degrees));
      }
    }

    SUBCASE("SCALAR_MULTIPLY") {
      ElementUnaryAttrs attrs = ElementUnaryAttrs{
          /*op_type=*/ElementUnaryOp::SCALAR_MULTIPLY,
          /*scalar=*/3.5f,
      };

      SUBCASE("data parallelism") {
        ParallelTensorDimDegrees input_degrees = mk_degrees(1, 1, 2, 1, 1);

        CHECK(element_unary_shard_signature_instance_is_valid(attrs,
                                                              input_degrees));
      }

      SUBCASE("inner dimension 1 parallelism") {
        ParallelTensorDimDegrees input_degrees = mk_degrees(1, 1, 1, 2, 1);

        CHECK(element_unary_shard_signature_instance_is_valid(attrs,
                                                              input_degrees));
      }

      SUBCASE("inner dimension 2 parallelism") {
        ParallelTensorDimDegrees input_degrees = mk_degrees(1, 1, 1, 1, 2);

        CHECK(element_unary_shard_signature_instance_is_valid(attrs,
                                                              input_degrees));
      }
    }
  }
}
