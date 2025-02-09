#include "op-attrs/ops/embedding.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "test/utils/doctest/fmt/expected.h"
#include "utils/integer_conversions.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("Sum embedding shape inference") {
    nonnegative_int out_channels = 128_n;
    nonnegative_int num_entries = 1024_n;
    EmbeddingAttrs attrs = EmbeddingAttrs{
        /*num_entries=*/num_entries,
        /*out_channels=*/out_channels,
        /*aggr=*/AggregateOp::SUM,
        /*data_type=*/DataType::FLOAT,
    };

    nonnegative_int batch_size = 48_n;
    nonnegative_int features_dim = 56_n;

    TensorShape input = TensorShape{
        TensorDims{FFOrdered<nonnegative_int>{
            batch_size,
            features_dim,
        }},
        DataType::INT32,
    };

    TensorShape output = TensorShape{
        TensorDims{
            FFOrdered<nonnegative_int>{
                batch_size,
                out_channels,
            },
        },
        DataType::FLOAT,
    };

    TensorShape weights = TensorShape{
        TensorDims{
            FFOrdered<nonnegative_int>{
                num_entries,
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

    // get_weights_shape
    {
      tl::expected<TensorShape, std::string> weight_result =
          get_weights_shape(attrs, input);
      tl::expected<TensorShape, std::string> weight_correct = weights;
      CHECK(weight_result == weight_correct);
    }

    auto make_input = [&](SumDegree o_sum,
                          DiscardCopyDegree o_eq,
                          nonnegative_int o_batch,
                          nonnegative_int o_features) {
      return lift_to_parallel_with_degrees(
          input, o_sum, o_eq, FFOrdered<nonnegative_int>{o_batch, o_features});
    };

    auto make_output = [&](SumDegree o_sum,
                           DiscardCopyDegree o_eq,
                           nonnegative_int o_batch,
                           nonnegative_int o_outchannels) {
      return lift_to_parallel_with_degrees(
          output,
          o_sum,
          o_eq,
          FFOrdered<nonnegative_int>{o_batch, o_outchannels});
    };

    auto make_weights = [&](SumDegree o_sum,
                            DiscardCopyDegree o_eq,
                            nonnegative_int o_entries,
                            nonnegative_int o_outchannels) {
      return lift_to_parallel_with_degrees(
          weights,
          o_sum,
          o_eq,
          FFOrdered<nonnegative_int>{o_entries, o_outchannels});
    };

    SUBCASE("data parallelism") {
      nonnegative_int degree = 4_n;
      ParallelTensorShape par_input =
          make_input(SumDegree{1_n}, DiscardCopyDegree{1_n}, degree, 1_n);

      {
        tl::expected<ParallelTensorShape, std::string> result =
            get_output_shape(attrs, par_input);
        tl::expected<ParallelTensorShape, std::string> correct =
            make_output(SumDegree{1_n}, DiscardCopyDegree{1_n}, degree, 1_n);
        CHECK(result == correct);
      }

      {
        tl::expected<ParallelTensorShape, std::string> result =
            get_weights_shape(attrs, par_input);
        tl::expected<ParallelTensorShape, std::string> correct =
            make_weights(SumDegree{1_n}, DiscardCopyDegree{degree}, 1_n, 1_n);
        CHECK(result == correct);
      }
    }

    SUBCASE("input features parallelism") {
      nonnegative_int degree = 4_n;
      ParallelTensorShape input =
          make_input(SumDegree{1_n}, DiscardCopyDegree{1_n}, 1_n, degree);

      {
        tl::expected<ParallelTensorShape, std::string> result =
            get_output_shape(attrs, input);
        tl::expected<ParallelTensorShape, std::string> correct =
            make_output(SumDegree{degree}, DiscardCopyDegree{1_n}, 1_n, 1_n);
        CHECK(result == correct);
      }

      {
        tl::expected<ParallelTensorShape, std::string> result =
            get_weights_shape(attrs, input);
        tl::expected<ParallelTensorShape, std::string> correct =
            make_weights(SumDegree{1_n}, DiscardCopyDegree{degree}, 1_n, 1_n);
        CHECK(result == correct);
      }
    }

    SUBCASE("output channel shard parallelism") {
      // NOTE (@lockshaw): in the current (parallel shape inference from just
      // input tensor) representation we have to choose between either
      // parallelism in the weight channel dimension or in the weight entry
      // dimension. For now we choose to represent parallelism in the channel
      // dimension, but partitioning in the entry dimension is also potentially
      // useful as it produces sum parallelism in the output
      nonnegative_int degree = 4_n;
      ParallelTensorShape input =
          make_input(SumDegree{1_n}, DiscardCopyDegree{degree}, 1_n, 1_n);

      {
        tl::expected<ParallelTensorShape, std::string> result =
            get_output_shape(attrs, input);
        tl::expected<ParallelTensorShape, std::string> correct =
            make_output(SumDegree{1_n}, DiscardCopyDegree{1_n}, 1_n, degree);
        CHECK(result == correct);
      }

      {
        tl::expected<ParallelTensorShape, std::string> result =
            get_weights_shape(attrs, input);
        tl::expected<ParallelTensorShape, std::string> correct =
            make_weights(SumDegree{1_n}, DiscardCopyDegree{1_n}, 1_n, degree);
        CHECK(result == correct);
      }
    }
  }
}
