#include "op-attrs/ops/linear.h"
#include "kernels/accessor.h"
#include "kernels/create_zero_filled_accessor.h"
#include "kernels/emulated_parallel_tensor/shard_signature_instance_is_valid.h"
#include "kernels/linear_kernels_cpu.h"
#include "kernels/local_cpu_allocator.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "test/utils/doctest/fmt/expected.h"
#include "utils/integer_conversions.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("get_linear_incoming_tensor_roles") {
    auto make_attrs = [](bool use_bias) {
      return LinearAttrs{
          /*out_channels=*/16_p,
          /*use_bias=*/use_bias,
          /*data_type=*/DataType::FLOAT,
          /*activation=*/Activation::RELU,
          /*regularizer=*/std::nullopt,
      };
    };

    SUBCASE("use_bias = true") {
      LinearAttrs attrs = make_attrs(/*use_bias=*/true);

      std::map<TensorSlotName, IncomingTensorRole> result =
          get_linear_incoming_tensor_roles(attrs);
      std::map<TensorSlotName, IncomingTensorRole> correct = {
          {
              TensorSlotName::INPUT,
              IncomingTensorRole::INPUT,
          },
          {
              TensorSlotName::WEIGHT,
              IncomingTensorRole::WEIGHT,
          },
          {
              TensorSlotName::BIAS,
              IncomingTensorRole::WEIGHT,
          },
      };

      CHECK(result == correct);
    }

    SUBCASE("use_bias = false") {
      LinearAttrs attrs = make_attrs(/*use_bias=*/false);

      std::map<TensorSlotName, IncomingTensorRole> result =
          get_linear_incoming_tensor_roles(attrs);
      std::map<TensorSlotName, IncomingTensorRole> correct = {
          {
              TensorSlotName::INPUT,
              IncomingTensorRole::INPUT,
          },
          {
              TensorSlotName::WEIGHT,
              IncomingTensorRole::WEIGHT,
          },
      };

      CHECK(result == correct);
    }
  }

  // TODO(@lockshaw): update this to a more modern test style
  TEST_CASE("Linear shape inference") {
    positive_int out_channels = 16_p;
    LinearAttrs attrs = LinearAttrs{
        /*out_channels=*/out_channels,
        /*use_bias=*/true,
        /*data_type=*/DataType::FLOAT,
        /*activation=*/Activation::RELU,
        /*regularizer=*/std::nullopt,
    };

    positive_int batch_size = 12_p;
    positive_int extra_dim = 16_p;
    positive_int in_channels = 8_p;

    TensorShape input = TensorShape{
        TensorDims{
            FFOrdered{
                batch_size,
                extra_dim,
                in_channels,
            },
        },
        DataType::FLOAT,
    };

    TensorShape output = TensorShape{
        TensorDims{
            FFOrdered{
                batch_size,
                extra_dim,
                out_channels,
            },
        },
        DataType::FLOAT,
    };

    TensorShape projection = TensorShape{
        TensorDims{
            FFOrdered{
                out_channels,
                in_channels,
            },
        },
        DataType::FLOAT,
    };

    TensorShape bias = TensorShape{
        TensorDims{
            FFOrdered{
                out_channels,
            },
        },
        DataType::FLOAT,
    };

    // linear_get_output_shape
    {
      TensorShape output_result = linear_get_output_shape(attrs, input);
      TensorShape output_correct = output;
      CHECK(output_result == output_correct);
    }

    // linear_get_weight_shape
    {
      TensorShape projection_result = linear_get_projection_shape(attrs, input);
      TensorShape projection_correct = projection;
      CHECK(projection_result == projection_correct);
    }

    // linear_get_bias_shape
    {
      TensorShape bias_result = linear_get_bias_shape(attrs, input);
      TensorShape bias_correct = bias;
      CHECK(bias_result == bias_correct);
    }

    auto make_input = [&](SumDegree o_sum,
                          DiscardCopyDegree o_eq,
                          positive_int o_batch,
                          positive_int o_extra_dim,
                          positive_int o_channel) {
      return lift_shape_to_parallel_with_degrees(
          input, o_sum, o_eq, FFOrdered{o_batch, o_extra_dim, o_channel});
    };

    auto make_output = [&](SumDegree o_sum,
                           DiscardCopyDegree o_eq,
                           positive_int o_batch,
                           positive_int o_extra_dim,
                           positive_int o_channel) {
      return lift_shape_to_parallel_with_degrees(
          output, o_sum, o_eq, FFOrdered{o_batch, o_extra_dim, o_channel});
    };

    auto make_projection = [&](SumDegree o_sum,
                               DiscardCopyDegree o_eq,
                               positive_int o_outchannel,
                               positive_int o_inchannel) {
      return lift_shape_to_parallel_with_degrees(
          projection, o_sum, o_eq, FFOrdered{o_outchannel, o_inchannel});
    };

    auto make_bias = [&](SumDegree o_sum,
                         DiscardCopyDegree o_eq,
                         positive_int o_outchannel) {
      return lift_shape_to_parallel_with_degrees(
          bias, o_sum, o_eq, FFOrdered{o_outchannel});
    };

    SUBCASE("data parallelism") {
      positive_int input_sum_degree = 2_p;
      positive_int extra_dim_degree = 8_p;
      positive_int degree = 4_p;

      ParallelTensorShape par_input = make_input(SumDegree{input_sum_degree},
                                                 DiscardCopyDegree{1_p},
                                                 degree,
                                                 extra_dim_degree,
                                                 1_p);

      {
        ParallelTensorShape result =
            linear_get_output_parallel_shape(attrs, par_input);
        ParallelTensorShape correct = make_output(SumDegree{input_sum_degree},
                                                  DiscardCopyDegree{1_p},
                                                  degree,
                                                  extra_dim_degree,
                                                  1_p);
        CHECK(result == correct);
      }

      {
        ParallelTensorShape result =
            linear_get_projection_parallel_shape(attrs, par_input);
        ParallelTensorShape correct = make_projection(
            SumDegree{1_p},
            DiscardCopyDegree{input_sum_degree * degree * extra_dim_degree},
            1_p,
            1_p);
        CHECK(result == correct);
      }

      {
        ParallelTensorShape result =
            linear_get_bias_parallel_shape(attrs, par_input);
        ParallelTensorShape correct =
            make_bias(SumDegree{input_sum_degree},
                      DiscardCopyDegree{degree * extra_dim_degree},
                      1_p);
        CHECK(result == correct);
      }
    }

    SUBCASE("reduction parallelism") {
      positive_int input_sum_degree = 2_p;
      positive_int degree = 4_p;

      ParallelTensorShape par_input = make_input(SumDegree{input_sum_degree},
                                                 DiscardCopyDegree{1_p},
                                                 1_p,
                                                 1_p,
                                                 degree);

      {
        ParallelTensorShape result =
            linear_get_output_parallel_shape(attrs, par_input);
        ParallelTensorShape correct =
            make_output(SumDegree{input_sum_degree * degree},
                        DiscardCopyDegree{1_p},
                        1_p,
                        1_p,
                        1_p);
        CHECK(result == correct);
      }

      {
        ParallelTensorShape result =
            linear_get_projection_parallel_shape(attrs, par_input);
        ParallelTensorShape correct = make_projection(
            SumDegree{1_p}, DiscardCopyDegree{input_sum_degree}, 1_p, degree);
        CHECK(result == correct);
      }

      {
        ParallelTensorShape result =
            linear_get_bias_parallel_shape(attrs, par_input);
        ParallelTensorShape correct = make_bias(
            SumDegree{input_sum_degree * degree}, DiscardCopyDegree{1_p}, 1_p);
        CHECK(result == correct);
      }
    }

    SUBCASE("output channel parallelism") {
      positive_int input_sum_degree = 2_p;
      positive_int degree = 4_p;

      ParallelTensorShape par_input = make_input(SumDegree{input_sum_degree},
                                                 DiscardCopyDegree{degree},
                                                 1_p,
                                                 1_p,
                                                 1_p);

      {
        ParallelTensorShape result =
            linear_get_output_parallel_shape(attrs, par_input);
        ParallelTensorShape correct = make_output(SumDegree{input_sum_degree},
                                                  DiscardCopyDegree{1_p},
                                                  1_p,
                                                  1_p,
                                                  degree);
        CHECK(result == correct);
      }

      {
        ParallelTensorShape result =
            linear_get_projection_parallel_shape(attrs, par_input);
        ParallelTensorShape correct = make_projection(
            SumDegree{1_p}, DiscardCopyDegree{input_sum_degree}, degree, 1_p);
        CHECK(result == correct);
      }

      {
        ParallelTensorShape result =
            linear_get_bias_parallel_shape(attrs, par_input);
        ParallelTensorShape correct = make_bias(
            SumDegree{input_sum_degree}, DiscardCopyDegree{1_p}, degree);
        CHECK(result == correct);
      }
    }
  }

  TEST_CASE("linear_get_task_group") {
    Allocator cpu_allocator = create_local_cpu_memory_allocator();

    LinearAttrs attrs = LinearAttrs{
        /*out_channels=*/2_p,
        /*use_bias=*/true,
        /*data_type=*/DataType::FLOAT,
        /*activation=*/Activation::RELU,
        /*regularier=*/std::nullopt,
    };

    TensorShape input_shape = TensorShape{
        TensorDims{
            FFOrdered{
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
            int channel_shard_degree) -> ParallelTensorDimDegrees {
      return ParallelTensorDimDegrees{
          /*sum_degree=*/SumDegree{positive_int{sum_degree}},
          /*discard_copy_degree=*/
          DiscardCopyDegree{positive_int{discard_copy_degree}},
          /*shard_degrees=*/
          FFOrdered{
              positive_int{batch_shard_degree},
              positive_int{channel_shard_degree},
          },
      };
    };

    auto run_linear = [&](std::map<TensorSlotName, GenericTensorAccessorR> const
                              &incoming_shards)
        -> std::map<TensorSlotName, GenericTensorAccessorR> {
      GenericTensorAccessorR input_shard =
          incoming_shards.at(TensorSlotName::INPUT);
      GenericTensorAccessorR projection_shard =
          incoming_shards.at(TensorSlotName::WEIGHT);
      GenericTensorAccessorR bias_shard =
          incoming_shards.at(TensorSlotName::BIAS);

      TensorShape output_shard_shape = linear_get_output_shape(
          attrs, get_tensor_shape_for_accessor_r(input_shard));
      GenericTensorAccessorW output_shard =
          create_zero_filled_accessor_w(output_shard_shape, cpu_allocator);

      linear_cpu_forward_kernel(
          /*attrs=*/attrs,
          /*input=*/input_shard,
          /*output=*/output_shard,
          /*projection=*/projection_shard,
          /*bias=*/bias_shard);

      return std::map<TensorSlotName, GenericTensorAccessorR>{
          {
              TensorSlotName::OUTPUT,
              read_only_accessor_from_write_accessor(output_shard),
          },
      };
    };

    auto linear_shard_signature_instance_is_valid =
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
          /*run_op=*/run_linear,
          /*seed=*/0);
    };

    SUBCASE("data parallelism") {
      ParallelTensorDimDegrees input_dim_degrees = mk_dim_degrees(1, 1, 2, 1);

      CHECK(linear_shard_signature_instance_is_valid(input_dim_degrees));
    }
  }
}
