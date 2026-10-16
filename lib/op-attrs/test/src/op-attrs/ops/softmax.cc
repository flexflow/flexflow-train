#include "op-attrs/ops/softmax.h"
#include "kernels/accessor.h"
#include "kernels/create_zero_filled_accessor.h"
#include "kernels/emulated_parallel_tensor/shard_signature_instance_is_valid.h"
#include "kernels/local_cpu_allocator.h"
#include "kernels/softmax_kernels_cpu.h"
#include "op-attrs/parallel_tensor_dims.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "test/utils/doctest/fmt/optional.h"
#include "utils/expected.h"
#include "utils/fmt/expected.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("softmax_get_output_shape") {
    TensorShape input = TensorShape{
        TensorDims{FFOrdered{
            12_p,
            14_p,
            16_p,
        }},
        DataType::FLOAT,
    };

    SUBCASE("attrs.dim in bounds") {
      SoftmaxAttrs attrs = SoftmaxAttrs{ff_dim_t{1_n}};

      TensorShape result = softmax_get_output_shape(attrs, input);
      TensorShape correct = input;

      CHECK(result == correct);
    }

    SUBCASE("attrs.dims out of bounds") {
      SoftmaxAttrs attrs = SoftmaxAttrs{ff_dim_t{4_n}};

      CHECK_THROWS(softmax_get_output_shape(attrs, input));
    }
  }

  TEST_CASE("softmax_get_output_parallel_shape") {
    TensorShape input = TensorShape{
        TensorDims{FFOrdered{
            12_p,
            14_p,
            16_p,
        }},
        DataType::FLOAT,
    };
    TensorShape output = input;

    auto make_input = [&](SumDegree o_sum,
                          DiscardCopyDegree o_eq,
                          positive_int o0,
                          positive_int o1,
                          positive_int o2) -> ParallelTensorShape {
      return lift_shape_to_parallel_with_degrees(
          input, o_sum, o_eq, FFOrdered{o0, o1, o2});
    };

    auto make_output = [&](SumDegree o_sum,
                           DiscardCopyDegree o_eq,
                           positive_int o0,
                           positive_int o1,
                           positive_int o2) -> ParallelTensorShape {
      return lift_shape_to_parallel_with_degrees(
          output, o_sum, o_eq, FFOrdered{o0, o1, o2});
    };

    SUBCASE("partition parallelism in non-softmax-dim (valid)") {
      positive_int degree0 = 2_p;
      positive_int degree2 = 4_p;

      ParallelTensorShape par_input = make_input(
          SumDegree{1_p}, DiscardCopyDegree{1_p}, degree0, 1_p, degree2);

      SUBCASE("attrs.dim in bounds") {
        SoftmaxAttrs attrs = SoftmaxAttrs{ff_dim_t{1_n}};

        ParallelTensorShape result =
            softmax_get_output_parallel_shape(attrs, par_input);

        ParallelTensorShape correct = make_output(
            SumDegree{1_p}, DiscardCopyDegree{1_p}, degree0, 1_p, degree2);

        CHECK(result == correct);
      }

      SUBCASE("attrs.dims out of bounds") {
        SoftmaxAttrs attrs = SoftmaxAttrs{ff_dim_t{4_n}};

        CHECK_THROWS(softmax_get_output_parallel_shape(attrs, par_input));
      }
    }

    SUBCASE("discard copy parallelism") {
      DiscardCopyDegree discard_copy_degree = DiscardCopyDegree{2_p};

      SoftmaxAttrs attrs = SoftmaxAttrs{ff_dim_t{1_n}};

      ParallelTensorShape par_input =
          make_input(SumDegree{1_p}, discard_copy_degree, 1_p, 1_p, 1_p);

      ParallelTensorShape result =
          softmax_get_output_parallel_shape(attrs, par_input);

      ParallelTensorShape correct =
          make_output(SumDegree{1_p}, discard_copy_degree, 1_p, 1_p, 1_p);

      CHECK(result == correct);
    }

    SUBCASE("partition parallism in softmax dim (invalid)") {
      positive_int degree1 = 2_p;

      SoftmaxAttrs attrs = SoftmaxAttrs{ff_dim_t{1_n}};

      ParallelTensorShape par_input =
          make_input(SumDegree{1_p}, DiscardCopyDegree{1_p}, 1_p, degree1, 1_p);

      CHECK_THROWS(softmax_get_output_parallel_shape(attrs, par_input));
    }

    SUBCASE("sum parallelism (invalid)") {
      SumDegree sum_degree = SumDegree{2_p};

      SoftmaxAttrs attrs = SoftmaxAttrs{ff_dim_t{1_n}};

      ParallelTensorShape par_input =
          make_input(sum_degree, DiscardCopyDegree{1_p}, 1_p, 1_p, 1_p);

      CHECK_THROWS(softmax_get_output_parallel_shape(attrs, par_input));
    }
  }

  TEST_CASE("softmax_get_task_group") {
    Allocator cpu_allocator = create_local_cpu_memory_allocator();

    SoftmaxAttrs attrs = SoftmaxAttrs{
        /*dim=*/ff_dim_t{1_n},
    };

    TensorShape input_shape = TensorShape{
        TensorDims{
            FFOrdered{
                6_p,
                4_p,
                8_p,
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

    auto run_softmax =
        [&](std::map<TensorSlotName, GenericTensorAccessorR> const
                &incoming_shards)
        -> std::map<TensorSlotName, GenericTensorAccessorR> {
      GenericTensorAccessorR input_shard =
          incoming_shards.at(TensorSlotName::INPUT);
      TensorShape output_shard_shape = softmax_get_output_shape(
          attrs, get_tensor_shape_for_accessor_r(input_shard));
      GenericTensorAccessorW output_shard =
          create_zero_filled_accessor_w(output_shard_shape, cpu_allocator);

      softmax_cpu_forward_kernel(
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

    auto softmax_shard_signature_instance_is_valid =
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
          /*run_op=*/run_softmax,
          /*seed=*/0);
    };

    SUBCASE("data parallelism") {
      ParallelTensorDimDegrees input_dim_degrees =
          mk_dim_degrees(1, 1, 2, 1, 1);

      CHECK(softmax_shard_signature_instance_is_valid(input_dim_degrees));
    }

    SUBCASE("non-dim parallelism") {
      ParallelTensorDimDegrees input_dim_degrees =
          mk_dim_degrees(1, 1, 1, 1, 2);

      CHECK(softmax_shard_signature_instance_is_valid(input_dim_degrees));
    }

    SUBCASE("discard copy parallelism") {
      ParallelTensorDimDegrees input_dim_degrees =
          mk_dim_degrees(1, 2, 1, 1, 1);

      CHECK(softmax_shard_signature_instance_is_valid(input_dim_degrees));
    }

    SUBCASE("hybrid parallelism") {
      ParallelTensorDimDegrees input_dim_degrees =
          mk_dim_degrees(1, 2, 2, 1, 2);

      CHECK(softmax_shard_signature_instance_is_valid(input_dim_degrees));
    }
  }
}
