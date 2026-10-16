#include "op-attrs/ops/element_binary.h"
#include "kernels/create_zero_filled_accessor.h"
#include "kernels/element_binary_kernels_cpu.h"
#include "kernels/emulated_parallel_tensor/shard_signature_instance_is_valid.h"
#include "kernels/local_cpu_allocator.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "op-attrs/tensor_dims.h"
#include "test/utils/doctest/fmt/expected.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("element_binary_get_output_shape") {
    positive_int d1 = 16_p;
    positive_int d2 = 32_p;
    positive_int d3 = 24_p;

    ElementBinaryAttrs attrs = ElementBinaryAttrs{
        ElementBinaryOp::ADD,
        DataType::FLOAT,
        /*should_broadcast_lhs=*/false,
        /*should_broadcast_rhs=*/false,
    };

    TensorShape input_lhs = TensorShape{
        TensorDims{
            FFOrdered{
                d1,
                d2,
                d3,
            },
        },
        DataType::FLOAT,
    };

    TensorShape input_rhs = input_lhs;

    SUBCASE("correct") {
      TensorShape result =
          element_binary_get_output_shape(attrs, input_lhs, input_rhs);
      TensorShape correct = input_lhs;

      CHECK(result == correct);
    }

    SUBCASE("mismatched dim size") {
      TensorShape incorrect_rhs = input_lhs;
      dim_at_idx(incorrect_rhs.dims, relative_ff_dim_t{0}) += 1_p;

      CHECK_THROWS(
          element_binary_get_output_shape(attrs, input_lhs, incorrect_rhs));
    }
  }

  TEST_CASE("element_binary_get_output_parallel_dim_degrees") {
    auto mk_attrs = [&](ElementBinaryOp op) -> ElementBinaryAttrs {
      return ElementBinaryAttrs{
          op,
          DataType::FLOAT,
          /*should_broadcast_lhs=*/false,
          /*should_broadcast_rhs=*/false,
      };
    };

    auto mk_3d_degrees = [&](int o_sum, int o_eq, int o_1, int o_2, int o_3)
        -> ParallelTensorDimDegrees {
      return ParallelTensorDimDegrees{
          /*sum_degree=*/SumDegree{positive_int{o_sum}},
          /*discard_copy_degree=*/DiscardCopyDegree{positive_int{o_eq}},
          /*shard_degrees=*/
          FFOrdered<positive_int>{
              positive_int{o_1},
              positive_int{o_2},
              positive_int{o_3},
          },
      };
    };

    SUBCASE("op = ADD") {
      ElementBinaryAttrs attrs = mk_attrs(ElementBinaryOp::ADD);

      SUBCASE("data parallelism") {
        ParallelTensorDimDegrees input_lhs = mk_3d_degrees(1, 1, 4, 1, 1);
        ParallelTensorDimDegrees input_rhs = mk_3d_degrees(1, 1, 4, 1, 1);

        ParallelTensorDimDegrees result =
            element_binary_get_output_parallel_dim_degrees(
                attrs, input_lhs, input_rhs);

        ParallelTensorDimDegrees correct = mk_3d_degrees(1, 1, 4, 1, 1);

        CHECK(result == correct);
      }

      SUBCASE("reduction parallelism") {
        ParallelTensorDimDegrees input_lhs = mk_3d_degrees(4, 1, 1, 1, 1);
        ParallelTensorDimDegrees input_rhs = mk_3d_degrees(4, 1, 1, 1, 1);

        ParallelTensorDimDegrees result =
            element_binary_get_output_parallel_dim_degrees(
                attrs, input_lhs, input_rhs);

        ParallelTensorDimDegrees correct = mk_3d_degrees(4, 1, 1, 1, 1);

        CHECK(result == correct);
      }

      SUBCASE("discard copy parallelism") {
        ParallelTensorDimDegrees input_lhs = mk_3d_degrees(1, 4, 1, 1, 1);
        ParallelTensorDimDegrees input_rhs = mk_3d_degrees(1, 4, 1, 1, 1);

        ParallelTensorDimDegrees result =
            element_binary_get_output_parallel_dim_degrees(
                attrs, input_lhs, input_rhs);

        ParallelTensorDimDegrees correct = mk_3d_degrees(1, 4, 1, 1, 1);

        CHECK(result == correct);
      }

      SUBCASE("invalid mismatched parallelism degrees") {
        ParallelTensorDimDegrees input_lhs = mk_3d_degrees(1, 1, 1, 4, 1);
        ParallelTensorDimDegrees input_rhs = mk_3d_degrees(1, 1, 1, 1, 4);

        CHECK_THROWS(element_binary_get_output_parallel_dim_degrees(
            attrs, input_lhs, input_rhs));
      }
    }
  }

  TEST_CASE("element_binary_get_output_parallel_shape") {
    // since most of the edge cases are already tested in
    // element_binary_get_output_shape and element_binary_get_output_parallel_dim_degrees,
    // here we just do a basic check that they compose

    auto mk_attrs = [&](ElementBinaryOp op) -> ElementBinaryAttrs {
      return ElementBinaryAttrs{
          op,
          DataType::FLOAT,
          /*should_broadcast_lhs=*/false,
          /*should_broadcast_rhs=*/false,
      };
    };

    SUBCASE("op = ADD") {
      ElementBinaryAttrs attrs = mk_attrs(ElementBinaryOp::ADD);

      ParallelTensorShape lhs_input_shape = ParallelTensorShape{
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
      ParallelTensorShape rhs_input_shape = lhs_input_shape;

      ParallelTensorShape result = element_binary_get_output_parallel_shape(
          attrs, lhs_input_shape, rhs_input_shape);

      ParallelTensorShape correct = ParallelTensorShape{
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

      CHECK(result == correct);
    }
  }

  TEST_CASE("element_binary_get_task_group") {
    Allocator cpu_allocator = create_local_cpu_memory_allocator();

    auto mk_degrees = [](int sum_degree,
                         int discard_copy_degree,
                         int batch_dim_degree,
                         int d1_degree,
                         int d2_degree) -> ParallelTensorDimDegrees {
      return ParallelTensorDimDegrees{
          /*sum_degree=*/SumDegree{positive_int{sum_degree}},
          /*discard_copy_degree=*/
          DiscardCopyDegree{positive_int{discard_copy_degree}},
          /*shard_degrees=*/
          FFOrdered<positive_int>{
              positive_int{batch_dim_degree},
              positive_int{d1_degree},
              positive_int{d2_degree},
          },
      };
    };

    TensorShape lhs_shard_shape = TensorShape{
        /*dims=*/TensorDims{
            FFOrdered<positive_int>{
                4_p,
                2_p,
                8_p,
            },
        },
        /*data_type=*/DataType::FLOAT,
    };

    TensorShape rhs_shard_shape = lhs_shard_shape;

    ElementBinaryAttrs attrs = ElementBinaryAttrs{
        /*op=*/ElementBinaryOp::ADD,
        /*compute_type=*/DataType::FLOAT,
        /*should_broadcast_lhs=*/false,
        /*should_broadcast_rhs=*/false,
    };

    auto run_element_binary =
        [&](std::map<TensorSlotName, GenericTensorAccessorR> const
                &input_shards)
        -> std::map<TensorSlotName, GenericTensorAccessorR> {
      GenericTensorAccessorR lhs_input_shard =
          input_shards.at(TensorSlotName::LHS_INPUT);
      GenericTensorAccessorR rhs_input_shard =
          input_shards.at(TensorSlotName::RHS_INPUT);
      TensorShape output_shard_shape = element_binary_get_output_shape(
          attrs, lhs_input_shard.shape, rhs_input_shard.shape);
      GenericTensorAccessorW output_shard =
          create_zero_filled_accessor_w(output_shard_shape, cpu_allocator);

      element_binary_cpu_forward_kernel(
          /*attrs=*/attrs,
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

    auto element_binary_shard_signature_instance_is_valid =
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
          /*attrs=*/ComputationGraphOpAttrs{attrs},
          /*input_shapes=*/input_shapes,
          /*run_op=*/run_element_binary,
          /*seed=*/0);
    };

    SUBCASE("data parallelism") {
      ParallelTensorDimDegrees lhs_degrees = mk_degrees(1, 1, 2, 1, 1);
      ParallelTensorDimDegrees rhs_degrees = mk_degrees(1, 1, 2, 1, 1);

      CHECK(element_binary_shard_signature_instance_is_valid(lhs_degrees,
                                                             rhs_degrees));
    }

    SUBCASE("inner dimension 1 parallelism") {
      ParallelTensorDimDegrees lhs_degrees = mk_degrees(1, 1, 1, 2, 1);
      ParallelTensorDimDegrees rhs_degrees = mk_degrees(1, 1, 1, 2, 1);

      CHECK(element_binary_shard_signature_instance_is_valid(lhs_degrees,
                                                             rhs_degrees));
    }

    SUBCASE("inner dimension 2 parallelism") {
      ParallelTensorDimDegrees lhs_degrees = mk_degrees(1, 1, 1, 1, 2);
      ParallelTensorDimDegrees rhs_degrees = mk_degrees(1, 1, 1, 1, 2);

      CHECK(element_binary_shard_signature_instance_is_valid(lhs_degrees,
                                                             rhs_degrees));
    }

    SUBCASE("parallelism in all shard dims") {
      ParallelTensorDimDegrees lhs_degrees = mk_degrees(1, 1, 2, 2, 2);
      ParallelTensorDimDegrees rhs_degrees = mk_degrees(1, 1, 2, 2, 2);

      CHECK(element_binary_shard_signature_instance_is_valid(lhs_degrees,
                                                             rhs_degrees));
    }
  }
}
