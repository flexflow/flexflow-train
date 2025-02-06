#include "op-attrs/ops/linear.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "test/utils/doctest/fmt/expected.h"
#include "utils/integer_conversions.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("get_linear_incoming_tensor_roles(LinearAttrs)") {
    auto make_attrs = [](bool use_bias) {
      return LinearAttrs{
          /*out_channels=*/16_n,
          /*use_bias=*/use_bias,
          /*data_type=*/DataType::FLOAT,
          /*activation=*/Activation::RELU,
          /*regularizer=*/std::nullopt,
      };
    };

    SUBCASE("use_bias = true") {
      LinearAttrs attrs = make_attrs(/*use_bias=*/true);

      std::vector<IncomingTensorRole> result =
          get_linear_incoming_tensor_roles(attrs);
      std::vector<IncomingTensorRole> correct = {
          IncomingTensorRole::INPUT,
          IncomingTensorRole::WEIGHT,
          IncomingTensorRole::WEIGHT,
      };

      CHECK(result == correct);
    }

    SUBCASE("use_bias = false") {
      LinearAttrs attrs = make_attrs(/*use_bias=*/false);

      std::vector<IncomingTensorRole> result =
          get_linear_incoming_tensor_roles(attrs);
      std::vector<IncomingTensorRole> correct = {
          IncomingTensorRole::INPUT,
          IncomingTensorRole::WEIGHT,
      };

      CHECK(result == correct);
    }
  }

  TEST_CASE("Linear shape inference") {
    nonnegative_int out_channels = 16_n;
    LinearAttrs attrs = LinearAttrs{
        /*out_channels=*/out_channels,
        /*use_bias=*/true,
        /*data_type=*/DataType::FLOAT,
        /*activation=*/Activation::RELU,
        /*regularizer=*/std::nullopt,
    };

    nonnegative_int batch_size = 12_n;
    nonnegative_int extra_dim = 16_n;
    nonnegative_int in_channels = 8_n;

    TensorShape input = TensorShape{
        TensorDims{
            FFOrdered<nonnegative_int>{
                batch_size,
                extra_dim,
                in_channels,
            },
        },
        DataType::FLOAT,
    };

    TensorShape output = TensorShape{
        TensorDims{
            FFOrdered<nonnegative_int>{
                batch_size,
                extra_dim,
                out_channels,
            },
        },
        DataType::FLOAT,
    };

    TensorShape projection = TensorShape{
        TensorDims{
            FFOrdered<nonnegative_int>{
                in_channels,
                out_channels,
            },
        },
        DataType::FLOAT,
    };

    TensorShape bias = TensorShape{
        TensorDims{
            FFOrdered<nonnegative_int>{
                out_channels,
            },
        },
        DataType::FLOAT,
    };

    // get_output_shape
    {
      tl::expected<TensorShape, std::string> output_result =
          get_output_shape(attrs, input);
      tl::expected<TensorShape, std::string> output_correct = output;
      CHECK(output_result == output_correct);
    }

    // get_weight_shape
    {
      tl::expected<TensorShape, std::string> projection_result =
          get_projection_shape(attrs, input);
      tl::expected<TensorShape, std::string> projection_correct = projection;
      CHECK(projection_result == projection_correct);
    }

    // get_bias_shape
    {
      tl::expected<TensorShape, std::string> bias_result =
          get_bias_shape(attrs, input);
      tl::expected<TensorShape, std::string> bias_correct = bias;
      CHECK(bias_result == bias_correct);
    }

    auto make_input = [&](SumDegree o_sum,
                          DiscardCopyDegree o_eq,
                          nonnegative_int o_batch,
                          nonnegative_int o_extra_dim,
                          nonnegative_int o_channel) {
      return lift_to_parallel_with_degrees(
          input,
          o_sum,
          o_eq,
          FFOrdered<nonnegative_int>{o_batch, o_extra_dim, o_channel});
    };

    auto make_output = [&](SumDegree o_sum,
                           DiscardCopyDegree o_eq,
                           nonnegative_int o_batch,
                           nonnegative_int o_extra_dim,
                           nonnegative_int o_channel) {
      return lift_to_parallel_with_degrees(
          output,
          o_sum,
          o_eq,
          FFOrdered<nonnegative_int>{o_batch, o_extra_dim, o_channel});
    };

    auto make_projection = [&](SumDegree o_sum,
                               DiscardCopyDegree o_eq,
                               nonnegative_int o_inchannel,
                               nonnegative_int o_outchannel) {
      return lift_to_parallel_with_degrees(
          projection,
          o_sum,
          o_eq,
          FFOrdered<nonnegative_int>{o_inchannel, o_outchannel});
    };

    auto make_bias = [&](SumDegree o_sum,
                         DiscardCopyDegree o_eq,
                         nonnegative_int o_outchannel) {
      return lift_to_parallel_with_degrees(
          bias, o_sum, o_eq, FFOrdered<nonnegative_int>{o_outchannel});
    };

    SUBCASE("data parallelism") {
      nonnegative_int input_sum_degree = 2_n;
      nonnegative_int extra_dim_degree = 8_n;
      nonnegative_int degree = 4_n;

      ParallelTensorShape par_input = make_input(SumDegree{input_sum_degree},
                                                 DiscardCopyDegree{1_n},
                                                 degree,
                                                 extra_dim_degree,
                                                 1_n);

      {
        tl::expected<ParallelTensorShape, std::string> result =
            get_output_shape(attrs, par_input);
        tl::expected<ParallelTensorShape, std::string> correct =
            make_output(SumDegree{input_sum_degree},
                        DiscardCopyDegree{1_n},
                        degree,
                        extra_dim_degree,
                        1_n);
        CHECK(result == correct);
      }

      {
        tl::expected<ParallelTensorShape, std::string> result =
            get_projection_shape(attrs, par_input);
        tl::expected<ParallelTensorShape, std::string> correct =
            make_projection(
                SumDegree{1_n},
                DiscardCopyDegree{input_sum_degree * degree * extra_dim_degree},
                1_n,
                1_n);
        CHECK(result == correct);
      }

      {
        tl::expected<ParallelTensorShape, std::string> result =
            get_bias_shape(attrs, par_input);
        tl::expected<ParallelTensorShape, std::string> correct =
            make_bias(SumDegree{input_sum_degree},
                      DiscardCopyDegree{degree * extra_dim_degree},
                      1_n);
        CHECK(result == correct);
      }
    }

    SUBCASE("reduction parallelism") {
      nonnegative_int input_sum_degree = 2_n;
      nonnegative_int degree = 4_n;

      ParallelTensorShape par_input = make_input(SumDegree{input_sum_degree},
                                                 DiscardCopyDegree{1_n},
                                                 1_n,
                                                 1_n,
                                                 degree);

      {
        tl::expected<ParallelTensorShape, std::string> result =
            get_output_shape(attrs, par_input);
        tl::expected<ParallelTensorShape, std::string> correct =
            make_output(SumDegree{input_sum_degree * degree},
                        DiscardCopyDegree{1_n},
                        1_n,
                        1_n,
                        1_n);
        CHECK(result == correct);
      }

      {
        tl::expected<ParallelTensorShape, std::string> result =
            get_projection_shape(attrs, par_input);
        tl::expected<ParallelTensorShape, std::string> correct =
            make_projection(SumDegree{1_n},
                            DiscardCopyDegree{input_sum_degree},
                            degree,
                            1_n);
        CHECK(result == correct);
      }

      {
        tl::expected<ParallelTensorShape, std::string> result =
            get_bias_shape(attrs, par_input);
        tl::expected<ParallelTensorShape, std::string> correct = make_bias(
            SumDegree{input_sum_degree * degree}, DiscardCopyDegree{1_n}, 1_n);
        CHECK(result == correct);
      }
    }

    SUBCASE("output channel parallelism") {
      nonnegative_int input_sum_degree = 2_n;
      nonnegative_int degree = 4_n;

      ParallelTensorShape par_input = make_input(SumDegree{input_sum_degree},
                                                 DiscardCopyDegree{degree},
                                                 1_n,
                                                 1_n,
                                                 1_n);

      {
        tl::expected<ParallelTensorShape, std::string> result =
            get_output_shape(attrs, par_input);
        tl::expected<ParallelTensorShape, std::string> correct =
            make_output(SumDegree{input_sum_degree},
                        DiscardCopyDegree{1_n},
                        1_n,
                        1_n,
                        degree);
        CHECK(result == correct);
      }

      {
        tl::expected<ParallelTensorShape, std::string> result =
            get_projection_shape(attrs, par_input);
        tl::expected<ParallelTensorShape, std::string> correct =
            make_projection(SumDegree{1_n},
                            DiscardCopyDegree{input_sum_degree},
                            1_n,
                            degree);
        CHECK(result == correct);
      }

      {
        tl::expected<ParallelTensorShape, std::string> result =
            get_bias_shape(attrs, par_input);
        tl::expected<ParallelTensorShape, std::string> correct = make_bias(
            SumDegree{input_sum_degree}, DiscardCopyDegree{1_n}, degree);
        CHECK(result == correct);
      }
    }
  }
}
