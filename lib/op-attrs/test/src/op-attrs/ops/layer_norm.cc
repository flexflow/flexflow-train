#include "op-attrs/ops/layer_norm.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "utils/expected.h"
#include "utils/fmt/expected.h"
#include "utils/fmt/optional.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("get_layer_norm_incoming_tensor_roles(LayerNormAttrs)") {
    auto make_attrs = [](bool elementwise_affine) {
      return LayerNormAttrs{
          /*axes=*/{ff_dim_t{nonnegative_int{0}}, ff_dim_t{nonnegative_int{2}}},
          elementwise_affine,
          /*eps=*/1.0,
      };
    };

    SUBCASE("elementwise_affine = true") {
      LayerNormAttrs attrs = make_attrs(/*elementwise_affine=*/true);

      std::vector<IncomingTensorRole> result =
          get_layer_norm_incoming_tensor_roles(attrs);
      std::vector<IncomingTensorRole> correct = {
          IncomingTensorRole::INPUT,
          IncomingTensorRole::WEIGHT,
          IncomingTensorRole::WEIGHT,
      };

      CHECK(result == correct);
    }

    SUBCASE("elementwise_affine = false") {
      LayerNormAttrs attrs = make_attrs(/*elementwise_affine=*/false);

      std::vector<IncomingTensorRole> result =
          get_layer_norm_incoming_tensor_roles(attrs);
      std::vector<IncomingTensorRole> correct = {
          IncomingTensorRole::INPUT,
      };

      CHECK(result == correct);
    }
  }

  TEST_CASE("shape inference (LayerNorm)") {
    LayerNormAttrs attrs_affine_true = LayerNormAttrs{
        /*axes=*/{ff_dim_t{nonnegative_int{1}}, ff_dim_t{nonnegative_int{3}}},
        /*elementwise_affine=*/true,
        /*eps=*/0.1,
    };

    LayerNormAttrs attrs_affine_false = [&] {
      LayerNormAttrs attrs = attrs_affine_true;
      attrs.elementwise_affine = false;
      return attrs;
    }();

    TensorShape input = TensorShape{
        TensorDims{FFOrdered<nonnegative_int>{
            12_n,
            14_n,
            16_n,
            18_n,
        }},
        DataType::FLOAT,
    };

    TensorShape output = input;

    TensorShape gamma = TensorShape{
        TensorDims{FFOrdered<nonnegative_int>{
            12_n,
            16_n,
        }},
        DataType::FLOAT,
    };

    TensorShape beta = gamma;

    SUBCASE("get_output_shape(LayerNormAttrs, TensorShape)") {
      tl::expected<TensorShape, std::string> result =
          get_output_shape(attrs_affine_true, input);
      tl::expected<TensorShape, std::string> correct = output;

      CHECK(result == correct);
    }

    SUBCASE("get_gamma_weights_shape(LayerNormAttrs, TensorShape)") {
      SUBCASE("elementwise_affine = true") {
        tl::expected<TensorShape, std::string> result =
            get_gamma_weights_shape(attrs_affine_true, input);
        tl::expected<TensorShape, std::string> correct = gamma;

        CHECK(result == correct);
      }

      SUBCASE("elementwise_affine = false") {
        std::optional<TensorShape> result = optional_from_expected(
            get_gamma_weights_shape(attrs_affine_false, input));
        std::optional<TensorShape> correct = std::nullopt;

        CHECK(result == correct);
      }
    }

    SUBCASE("get_beta_weights_shape(LayerNormAttrs, TensorShape)") {
      SUBCASE("elementwise_affine = true") {
        tl::expected<TensorShape, std::string> result =
            get_beta_weights_shape(attrs_affine_true, input);
        tl::expected<TensorShape, std::string> correct = beta;

        CHECK(result == correct);
      }

      SUBCASE("elementwise_affine = false") {
        std::optional<TensorShape> result = optional_from_expected(
            get_beta_weights_shape(attrs_affine_false, input));
        std::optional<TensorShape> correct = std::nullopt;

        CHECK(result == correct);
      }
    }

    auto make_input = [&](SumDegree o_sum,
                          DiscardCopyDegree o_eq,
                          nonnegative_int o0,
                          nonnegative_int o1,
                          nonnegative_int o2,
                          nonnegative_int o3) {
      return lift_to_parallel_with_degrees(
          input, o_sum, o_eq, FFOrdered<nonnegative_int>{o0, o1, o2, o3});
    };

    auto make_output = [&](SumDegree o_sum,
                           DiscardCopyDegree o_eq,
                           nonnegative_int o0,
                           nonnegative_int o1,
                           nonnegative_int o2,
                           nonnegative_int o3) {
      return lift_to_parallel_with_degrees(
          output, o_sum, o_eq, FFOrdered<nonnegative_int>{o0, o1, o2, o3});
    };

    auto make_gamma_weights = [&](SumDegree o_sum,
                                  DiscardCopyDegree o_eq,
                                  nonnegative_int o0,
                                  nonnegative_int o2) {
      return lift_to_parallel_with_degrees(
          gamma, o_sum, o_eq, FFOrdered<nonnegative_int>{o0, o2});
    };

    auto make_beta_weights = [&](SumDegree o_sum,
                                 DiscardCopyDegree o_eq,
                                 nonnegative_int o0,
                                 nonnegative_int o2) {
      return lift_to_parallel_with_degrees(
          beta, o_sum, o_eq, FFOrdered<nonnegative_int>{o0, o2});
    };

    SUBCASE("parallel shape inference (LayerNorm)") {
      SUBCASE("partition parallelism (not in axes)") {
        nonnegative_int degree0 = 2_n;
        nonnegative_int degree2 = 3_n;

        ParallelTensorShape par_input = make_input(
            SumDegree{1_n}, DiscardCopyDegree{1_n}, degree0, 1_n, degree2, 1_n);

        SUBCASE("get_output_shape(LayerNormAttrs, ParallelTensorShape)") {
          tl::expected<ParallelTensorShape, std::string> result =
              get_output_shape(attrs_affine_true, par_input);
          tl::expected<ParallelTensorShape, std::string> correct =
              make_output(SumDegree{1_n},
                          DiscardCopyDegree{1_n},
                          degree0,
                          1_n,
                          degree2,
                          1_n);

          CHECK(result == correct);
        }

        SUBCASE(
            "get_gamma_weights_shape(LayerNormAttrs, ParallelTensorShape)") {
          SUBCASE("elementwise_affine = true") {
            tl::expected<ParallelTensorShape, std::string> result =
                get_gamma_weights_shape(attrs_affine_true, par_input);
            tl::expected<ParallelTensorShape, std::string> correct =
                make_gamma_weights(
                    SumDegree{1_n}, DiscardCopyDegree{1_n}, degree0, degree2);

            CHECK(result == correct);
          }

          SUBCASE("elementwise_affine = false") {
            std::optional<ParallelTensorShape> result = optional_from_expected(
                get_gamma_weights_shape(attrs_affine_false, par_input));
            std::optional<ParallelTensorShape> correct = std::nullopt;

            CHECK(result == correct);
          }
        }

        SUBCASE("get_beta_weights_shape(LayerNormAttrs, ParallelTensorShape)") {
          SUBCASE("elementwise_affine = true") {
            tl::expected<ParallelTensorShape, std::string> result =
                get_beta_weights_shape(attrs_affine_true, par_input);
            tl::expected<ParallelTensorShape, std::string> correct =
                make_beta_weights(
                    SumDegree{1_n}, DiscardCopyDegree{1_n}, degree0, degree2);

            CHECK(result == correct);
          }

          SUBCASE("elementwise_affine = false") {
            std::optional<ParallelTensorShape> result = optional_from_expected(
                get_beta_weights_shape(attrs_affine_false, par_input));
            std::optional<ParallelTensorShape> correct = std::nullopt;

            CHECK(result == correct);
          }
        }
      }

      SUBCASE("partition parallelism (in axes)") {
        nonnegative_int degree1 = 2_n;
        nonnegative_int degree2 = 4_n;

        ParallelTensorShape par_input = make_input(
            SumDegree{1_n}, DiscardCopyDegree{1_n}, 1_n, degree1, degree2, 1_n);

        SUBCASE("get_output_shape(LayerNormAttrs, ParallelTensorShape)") {
          std::optional<ParallelTensorShape> result = optional_from_expected(
              get_output_shape(attrs_affine_true, par_input));
          std::optional<ParallelTensorShape> correct = std::nullopt;

          CHECK(result == correct);
        }

        SUBCASE(
            "get_gamma_weights_shape(LayerNormAttrs, ParallelTensorShape)") {
          std::optional<ParallelTensorShape> result = optional_from_expected(
              get_gamma_weights_shape(attrs_affine_true, par_input));
          std::optional<ParallelTensorShape> correct = std::nullopt;

          CHECK(result == correct);
        }

        SUBCASE("get_beta_weights_shape(LayerNormAttrs, ParallelTensorShape)") {
          std::optional<ParallelTensorShape> result = optional_from_expected(
              get_beta_weights_shape(attrs_affine_true, par_input));
          std::optional<ParallelTensorShape> correct = std::nullopt;

          CHECK(result == correct);
        }
      }

      SUBCASE("sum parallelism") {
        SumDegree sum_degree = SumDegree{2_n};

        ParallelTensorShape par_input =
            make_input(sum_degree, DiscardCopyDegree{1_n}, 1_n, 1_n, 1_n, 1_n);

        SUBCASE("get_output_shape(LayerNormAttrs, ParallelTensorShape)") {
          std::optional<ParallelTensorShape> result = optional_from_expected(
              get_output_shape(attrs_affine_true, par_input));
          std::optional<ParallelTensorShape> correct = std::nullopt;

          CHECK(result == correct);
        }

        SUBCASE(
            "get_gamma_weights_shape(LayerNormAttrs, ParallelTensorShape)") {
          std::optional<ParallelTensorShape> result = optional_from_expected(
              get_gamma_weights_shape(attrs_affine_true, par_input));
          std::optional<ParallelTensorShape> correct = std::nullopt;

          CHECK(result == correct);
        }

        SUBCASE("get_beta_weights_shape(LayerNormAttrs, ParallelTensorShape)") {
          std::optional<ParallelTensorShape> result = optional_from_expected(
              get_beta_weights_shape(attrs_affine_true, par_input));
          std::optional<ParallelTensorShape> correct = std::nullopt;

          CHECK(result == correct);
        }
      }

      SUBCASE("discard copy parallelism") {
        DiscardCopyDegree discard_copy_degree = DiscardCopyDegree{2_n};

        ParallelTensorShape par_input =
            make_input(SumDegree{1_n}, discard_copy_degree, 1_n, 1_n, 1_n, 1_n);

        SUBCASE("get_output_shape(LayerNormAttrs, ParallelTensorShape)") {
          std::optional<ParallelTensorShape> result = optional_from_expected(
              get_output_shape(attrs_affine_true, par_input));
          std::optional<ParallelTensorShape> correct = std::nullopt;

          CHECK(result == correct);
        }

        SUBCASE(
            "get_gamma_weights_shape(LayerNormAttrs, ParallelTensorShape)") {
          std::optional<ParallelTensorShape> result = optional_from_expected(
              get_gamma_weights_shape(attrs_affine_true, par_input));
          std::optional<ParallelTensorShape> correct = std::nullopt;

          CHECK(result == correct);
        }

        SUBCASE("get_beta_weights_shape(LayerNormAttrs, ParallelTensorShape)") {
          std::optional<ParallelTensorShape> result = optional_from_expected(
              get_beta_weights_shape(attrs_affine_true, par_input));
          std::optional<ParallelTensorShape> correct = std::nullopt;

          CHECK(result == correct);
        }
      }
    }
  }
}
