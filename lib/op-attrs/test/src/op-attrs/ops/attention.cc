#include "op-attrs/ops/attention.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "test/utils/doctest/fmt/expected.h"
#include "utils/integer_conversions.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("get_attention_incoming_tensor_roles(MultiHeadAttentionAttrs)") {
    auto make_attrs = [](bool bias) {
      return MultiHeadAttentionAttrs{
          /*embed_dim=*/32_n,
          /*num_heads=*/10_n,
          /*kdim=*/32_n,
          /*vdim=*/32_n,
          /*dropout=*/0.0,
          /*bias=*/bias,
          /*add_bias_kv=*/false,
          /*add_zero_attn=*/false,
      };
    };

    SUBCASE("without bias") {
      MultiHeadAttentionAttrs attrs = make_attrs(/*bias=*/false);

      tl::expected<std::vector<IncomingTensorRole>, std::string> result =
          get_attention_incoming_tensor_roles(attrs);
      tl::expected<std::vector<IncomingTensorRole>, std::string> correct =
          std::vector{
              IncomingTensorRole::INPUT,
              IncomingTensorRole::INPUT,
              IncomingTensorRole::INPUT,
              IncomingTensorRole::WEIGHT,
          };

      CHECK(result == correct);
    }

    SUBCASE("with bias") {
      MultiHeadAttentionAttrs attrs = make_attrs(/*bias=*/true);

      tl::expected<std::vector<IncomingTensorRole>, std::string> result =
          get_attention_incoming_tensor_roles(attrs);
      tl::expected<std::vector<IncomingTensorRole>, std::string> correct =
          std::vector{
              IncomingTensorRole::INPUT,
              IncomingTensorRole::INPUT,
              IncomingTensorRole::INPUT,
              IncomingTensorRole::WEIGHT,
              IncomingTensorRole::WEIGHT,
              IncomingTensorRole::WEIGHT,
          };

      CHECK(result == correct);
    }
  }

  TEST_CASE("get_output_shape(MultiHeadAttentionAttrs, TensorShape, "
            "TensorShape, TensorShape)") {
    nonnegative_int embed_dim = 32_n;
    nonnegative_int num_heads = 10_n;

    /* Parameter meanings match those at
     * https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html
     */
    MultiHeadAttentionAttrs attrs = MultiHeadAttentionAttrs{
        /*embed_dim=*/embed_dim,
        /*num_heads=*/num_heads,
        /*kdim=*/embed_dim,
        /*vdim=*/embed_dim,
        /*dropout=*/0.0,
        /*bias=*/true,
        /*add_bias_kv=*/false,
        /*add_zero_attn=*/false,
    };

    nonnegative_int batch_size = 40_n;
    nonnegative_int seq_len = 48_n;
    nonnegative_int feature_size = 36_n;

    TensorShape input_q = TensorShape{
        TensorDims{
            FFOrdered<nonnegative_int>{
                batch_size,
                seq_len,
                feature_size,
            },
        },
        DataType::FLOAT,
    };

    TensorShape input_k = TensorShape{
        TensorDims{
            FFOrdered<nonnegative_int>{
                batch_size,
                seq_len,
                feature_size,
            },
        },
        DataType::FLOAT,
    };

    TensorShape input_v = TensorShape{
        TensorDims{
            FFOrdered<nonnegative_int>{
                batch_size,
                seq_len,
                feature_size,
            },
        },
        DataType::FLOAT,
    };

    TensorShape output = TensorShape{
        TensorDims{
            FFOrdered<nonnegative_int>{
                batch_size,
                seq_len,
                attrs.embed_dim,
            },
        },
        DataType::FLOAT,
    };

    TensorShape weights = TensorShape{
        TensorDims{
            FFOrdered<nonnegative_int>{
                (feature_size * embed_dim) * 3_n + (embed_dim * embed_dim),
                num_heads,
            },
        },
        DataType::FLOAT,
    };

    TensorShape input_bias = TensorShape{
        TensorDims{
            FFOrdered<nonnegative_int>{
                embed_dim * 3_n,
            },
        },
        DataType::FLOAT,
    };

    TensorShape output_bias = TensorShape{
        TensorDims{
            FFOrdered<nonnegative_int>{
                embed_dim,
            },
        },
        DataType::FLOAT,
    };

    SUBCASE("get_output_shape") {
      tl::expected<TensorShape, std::string> result =
          get_output_shape(attrs, input_q, input_k, input_v);

      tl::expected<TensorShape, std::string> correct = output;
      CHECK(result == correct);
    }

    SUBCASE("get_weights_shape") {
      tl::expected<TensorShape, std::string> result =
          get_weights_shape(attrs, input_q, input_k, input_v);

      tl::expected<TensorShape, std::string> correct = weights;
      CHECK(result == correct);
    }

    SUBCASE("get_input_bias_shape") {
      tl::expected<TensorShape, std::string> result =
          get_input_bias_shape(attrs, input_q, input_k, input_v);
      tl::expected<TensorShape, std::string> correct = input_bias;
      CHECK(result == correct);
    }

    SUBCASE("get_output_bias_shape") {
      tl::expected<TensorShape, std::string> result =
          get_output_bias_shape(attrs, input_q, input_k, input_v);
      tl::expected<TensorShape, std::string> correct = output_bias;
      CHECK(result == correct);
    }

    SUBCASE("parallel shape inference") {
      auto make_q = [&](SumDegree o_sum,
                        DiscardCopyDegree o_eq,
                        nonnegative_int o_batch,
                        nonnegative_int o_seq_len,
                        nonnegative_int o_q) {
        return lift_to_parallel_with_degrees(
            input_q,
            o_sum,
            o_eq,
            FFOrdered<nonnegative_int>{o_batch, o_seq_len, o_q});
      };

      auto make_k = [&](SumDegree o_sum,
                        DiscardCopyDegree o_eq,
                        nonnegative_int o_batch,
                        nonnegative_int o_seq_len,
                        nonnegative_int o_k) {
        return lift_to_parallel_with_degrees(
            input_k,
            o_sum,
            o_eq,
            FFOrdered<nonnegative_int>{o_batch, o_seq_len, o_k});
      };

      auto make_v = [&](SumDegree o_sum,
                        DiscardCopyDegree o_eq,
                        nonnegative_int o_batch,
                        nonnegative_int o_seq_len,
                        nonnegative_int o_v) {
        return lift_to_parallel_with_degrees(
            input_v,
            o_sum,
            o_eq,
            FFOrdered<nonnegative_int>{o_batch, o_seq_len, o_v});
      };

      auto make_o = [&](SumDegree o_sum,
                        DiscardCopyDegree o_eq,
                        nonnegative_int o_batch,
                        nonnegative_int o_seq_len,
                        nonnegative_int o_o) {
        return lift_to_parallel_with_degrees(
            output,
            o_sum,
            o_eq,
            FFOrdered<nonnegative_int>{o_batch, o_seq_len, o_o});
      };

      auto make_w = [&](SumDegree o_sum,
                        DiscardCopyDegree o_eq,
                        nonnegative_int o_e,
                        nonnegative_int o_h) {
        return lift_to_parallel_with_degrees(
            weights, o_sum, o_eq, FFOrdered<nonnegative_int>{o_e, o_h});
      };

      auto make_input_bias = [&](SumDegree o_sum,
                                 DiscardCopyDegree o_eq,
                                 nonnegative_int o_in_proj_channel) {
        return lift_to_parallel_with_degrees(
            input_bias,
            o_sum,
            o_eq,
            FFOrdered<nonnegative_int>{o_in_proj_channel});
      };

      auto make_output_bias = [&](SumDegree o_sum,
                                  DiscardCopyDegree o_eq,
                                  nonnegative_int o_out_proj_channel) {
        return lift_to_parallel_with_degrees(
            output_bias,
            o_sum,
            o_eq,
            FFOrdered<nonnegative_int>{o_out_proj_channel});
      };

      SUBCASE("data parallelism") {
        nonnegative_int o_b = 4_n;
        ParallelTensorShape q =
            make_q(SumDegree{1_n}, DiscardCopyDegree{1_n}, o_b, 1_n, 1_n);
        ParallelTensorShape k =
            make_k(SumDegree{1_n}, DiscardCopyDegree{1_n}, o_b, 1_n, 1_n);
        ParallelTensorShape v =
            make_v(SumDegree{1_n}, DiscardCopyDegree{1_n}, o_b, 1_n, 1_n);

        SUBCASE("get_output_shape") {
          tl::expected<ParallelTensorShape, std::string> result =
              get_output_shape(attrs, q, k, v);
          tl::expected<ParallelTensorShape, std::string> correct =
              make_o(SumDegree{1_n}, DiscardCopyDegree{1_n}, o_b, 1_n, 1_n);
          CHECK(result == correct);
        }

        SUBCASE("get_weights_shape") {
          tl::expected<ParallelTensorShape, std::string> result =
              get_weights_shape(attrs, q, k, v);
          tl::expected<ParallelTensorShape, std::string> correct =
              make_w(SumDegree{1_n}, DiscardCopyDegree{o_b}, 1_n, 1_n);
          CHECK(result == correct);
        }

        SUBCASE("get_input_bias_shape") {
          tl::expected<ParallelTensorShape, std::string> result =
              get_input_bias_shape(attrs, q, k, v);
          tl::expected<ParallelTensorShape, std::string> correct =
              make_input_bias(SumDegree{1_n}, DiscardCopyDegree{o_b}, 1_n);
          CHECK(result == correct);
        }

        SUBCASE("get_output_bias_shape") {
          tl::expected<ParallelTensorShape, std::string> result =
              get_output_bias_shape(attrs, q, k, v);
          tl::expected<ParallelTensorShape, std::string> correct =
              make_output_bias(SumDegree{1_n}, DiscardCopyDegree{o_b}, 1_n);
          CHECK(result == correct);
        }
      }

      SUBCASE("attention head parallelism") {
        nonnegative_int o_h = 2_n;
        ParallelTensorShape q =
            make_q(SumDegree{1_n}, DiscardCopyDegree{o_h}, 1_n, 1_n, 1_n);
        ParallelTensorShape k =
            make_k(SumDegree{1_n}, DiscardCopyDegree{o_h}, 1_n, 1_n, 1_n);
        ParallelTensorShape v =
            make_v(SumDegree{1_n}, DiscardCopyDegree{o_h}, 1_n, 1_n, 1_n);

        SUBCASE("get_output_shape") {
          tl::expected<ParallelTensorShape, std::string> result =
              get_output_shape(attrs, q, k, v);
          tl::expected<ParallelTensorShape, std::string> correct =
              make_o(SumDegree{o_h}, DiscardCopyDegree{1_n}, 1_n, 1_n, 1_n);
          CHECK(result == correct);
        }

        SUBCASE("get_weight_shape") {
          tl::expected<ParallelTensorShape, std::string> result =
              get_weights_shape(attrs, q, k, v);
          tl::expected<ParallelTensorShape, std::string> correct =
              make_w(SumDegree{1_n}, DiscardCopyDegree{1_n}, 1_n, o_h);
          CHECK(result == correct);
        }

        SUBCASE("get_input_bias_shape") {
          tl::expected<ParallelTensorShape, std::string> result =
              get_input_bias_shape(attrs, q, k, v);
          tl::expected<ParallelTensorShape, std::string> correct =
              make_input_bias(SumDegree{1_n}, DiscardCopyDegree{o_h}, 1_n);
          CHECK(result == correct);
        }

        SUBCASE("get_output_bias_shape") {
          tl::expected<ParallelTensorShape, std::string> result =
              get_output_bias_shape(attrs, q, k, v);
          tl::expected<ParallelTensorShape, std::string> correct =
              make_output_bias(SumDegree{1_n}, DiscardCopyDegree{o_h}, 1_n);
          CHECK(result == correct);
        }
      }

      SUBCASE("combined data & attention head parallelism") {
        nonnegative_int o_b = 4_n;
        nonnegative_int o_h = 2_n;
        ParallelTensorShape q =
            make_q(SumDegree{1_n}, DiscardCopyDegree{o_h}, o_b, 1_n, 1_n);
        ParallelTensorShape k =
            make_k(SumDegree{1_n}, DiscardCopyDegree{o_h}, o_b, 1_n, 1_n);
        ParallelTensorShape v =
            make_v(SumDegree{1_n}, DiscardCopyDegree{o_h}, o_b, 1_n, 1_n);

        SUBCASE("get_output_shape") {
          tl::expected<ParallelTensorShape, std::string> result =
              get_output_shape(attrs, q, k, v);
          tl::expected<ParallelTensorShape, std::string> correct =
              make_o(SumDegree{o_h}, DiscardCopyDegree{1_n}, o_b, 1_n, 1_n);
          CHECK(result == correct);
        }

        SUBCASE("get_weights_shape") {
          tl::expected<ParallelTensorShape, std::string> result =
              get_weights_shape(attrs, q, k, v);
          tl::expected<ParallelTensorShape, std::string> correct =
              make_w(SumDegree{1_n}, DiscardCopyDegree{o_b}, 1_n, o_h);
          CHECK(result == correct);
        }

        SUBCASE("get_input_bias_shape") {
          tl::expected<ParallelTensorShape, std::string> result =
              get_input_bias_shape(attrs, q, k, v);
          tl::expected<ParallelTensorShape, std::string> correct =
              make_input_bias(
                  SumDegree{1_n}, DiscardCopyDegree{o_b * o_h}, 1_n);
          CHECK(result == correct);
        }

        SUBCASE("get_output_bias_shape") {
          tl::expected<ParallelTensorShape, std::string> result =
              get_output_bias_shape(attrs, q, k, v);
          tl::expected<ParallelTensorShape, std::string> correct =
              make_output_bias(
                  SumDegree{1_n}, DiscardCopyDegree{o_b * o_h}, 1_n);
          CHECK(result == correct);
        }
      }
    }
  }
}
