#include "op-attrs/ops/batch_norm.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "utils/expected.h"
#include "utils/fmt/expected.h"
#include "utils/fmt/optional.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("get_batch_norm_incoming_tensor_roles") {
    auto make_attrs = [](bool affine) {
      return BatchNormAttrs{
          /*activation=*/std::nullopt,
          /*affine=*/affine,
          /*eps=*/1.0,
          /*momentum=*/0.1,
      };
    };

    SUBCASE("affine = true") {
      BatchNormAttrs attrs = make_attrs(/*affine=*/true);

      std::map<TensorSlotName, IncomingTensorRole> result =
          get_batch_norm_incoming_tensor_roles(attrs);
      std::map<TensorSlotName, IncomingTensorRole> correct = {
          {
              TensorSlotName::INPUT,
              IncomingTensorRole::INPUT,
          },
          {
              TensorSlotName::GAMMA,
              IncomingTensorRole::WEIGHT,
          },
          {
              TensorSlotName::BETA,
              IncomingTensorRole::WEIGHT,
          },
      };

      CHECK(result == correct);
    }

    SUBCASE("affine = false") {
      BatchNormAttrs attrs = make_attrs(/*affine=*/false);

      std::map<TensorSlotName, IncomingTensorRole> result =
          get_batch_norm_incoming_tensor_roles(attrs);
      std::map<TensorSlotName, IncomingTensorRole> correct = {
          {
              TensorSlotName::INPUT,
              IncomingTensorRole::INPUT,
          },
      };

      CHECK(result == correct);
    }
  }

  TEST_CASE("shape inference (BatchNorm)") {
    BatchNormAttrs attrs_affine_true = BatchNormAttrs{
        /*activation=*/std::nullopt,
        /*affine=*/true,
        /*eps=*/1.0,
        /*momentum=*/0.1,
    };

    BatchNormAttrs attrs_affine_false = [&] {
      BatchNormAttrs attrs = attrs_affine_true;
      attrs.affine = false;
      return attrs;
    }();

    TensorShape input = TensorShape{
        TensorDims{FFOrdered{
            12_p,
            14_p,
            16_p,
            18_p,
        }},
        DataType::FLOAT,
    };

    TensorShape output = input;

    TensorShape gamma = TensorShape{
        TensorDims{FFOrdered{
            14_p,
        }},
        DataType::FLOAT,
    };

    TensorShape beta = gamma;

    SUBCASE("batch_norm_get_output_shape") {
      TensorShape result =
          batch_norm_get_output_shape(attrs_affine_true, input);
      TensorShape correct = output;

      CHECK(result == correct);
    }

    SUBCASE("batch_norm_get_gamma_weights_shape") {
      SUBCASE("affine = true") {
        TensorShape result =
            batch_norm_get_gamma_weights_shape(attrs_affine_true, input);
        TensorShape correct = gamma;

        CHECK(result == correct);
      }

      SUBCASE("affine = false") {
        CHECK_THROWS(
            batch_norm_get_gamma_weights_shape(attrs_affine_false, input));
      }
    }

    SUBCASE("batch_norm_get_beta_weights_shape") {
      SUBCASE("affine = true") {
        TensorShape result =
            batch_norm_get_beta_weights_shape(attrs_affine_true, input);
        TensorShape correct = beta;

        CHECK(result == correct);
      }

      SUBCASE("affine = false") {
        CHECK_THROWS(
            batch_norm_get_beta_weights_shape(attrs_affine_false, input));
      }
    }
  }

  TEST_CASE("parallel dim degree inference (BatchNormAttrs)") {
    BatchNormAttrs attrs_affine_true = BatchNormAttrs{
        /*activation=*/std::nullopt,
        /*affine=*/true,
        /*eps=*/1.0,
        /*momentum=*/0.1,
    };

    BatchNormAttrs attrs_affine_false = [&] {
      BatchNormAttrs attrs = attrs_affine_true;
      attrs.affine = false;
      return attrs;
    }();

    SUBCASE("partition parallelism (in channel dim)") {
      positive_int degree = 2_p;

      ParallelTensorDimDegrees input = ParallelTensorDimDegrees{
          SumDegree{1_p},
          DiscardCopyDegree{1_p},
          FFOrdered{
              1_p,
              degree,
              1_p,
              1_p,
          },
      };

      SUBCASE("batch_norm_get_output_parallel_dim_degrees") {
        ParallelTensorDimDegrees result =
            batch_norm_get_output_parallel_dim_degrees(attrs_affine_true,
                                                       input);
        ParallelTensorDimDegrees correct = input;

        CHECK(result == correct);
      }

      SUBCASE("batch_norm_get_gamma_weights_parallel_dim_degrees") {
        SUBCASE("affine = true") {
          ParallelTensorDimDegrees result =
              batch_norm_get_gamma_weights_parallel_dim_degrees(
                  attrs_affine_true, input);
          ParallelTensorDimDegrees correct = ParallelTensorDimDegrees{
              SumDegree{1_p},
              DiscardCopyDegree{1_p},
              FFOrdered{degree},
          };

          CHECK(result == correct);
        }

        SUBCASE("affine = false") {
          CHECK_THROWS(batch_norm_get_gamma_weights_parallel_dim_degrees(
              attrs_affine_false, input));
        }
      }

      SUBCASE("batch_norm_get_beta_weights_parallel_dim_degrees") {
        SUBCASE("affine = true") {
          ParallelTensorDimDegrees result =
              batch_norm_get_beta_weights_parallel_dim_degrees(
                  attrs_affine_true, input);
          ParallelTensorDimDegrees correct = ParallelTensorDimDegrees{
              SumDegree{1_p},
              DiscardCopyDegree{1_p},
              FFOrdered{degree},
          };

          CHECK(result == correct);
        }

        SUBCASE("affine = false") {
          CHECK_THROWS(batch_norm_get_beta_weights_parallel_dim_degrees(
              attrs_affine_false, input));
        }
      }
    }

    SUBCASE("partition parallelism (not in channel dim)") {
      positive_int degree = 2_p;

      ParallelTensorDimDegrees input = ParallelTensorDimDegrees{
          SumDegree{1_p},
          DiscardCopyDegree{1_p},
          FFOrdered{1_p, 1_p, degree, 1_p},
      };

      SUBCASE("batch_norm_get_output_parallel_dim_degrees") {
        CHECK_THROWS(batch_norm_get_output_parallel_dim_degrees(
            attrs_affine_true, input));
      }

      SUBCASE("batch_norm_get_gamma_weights_parallel_dim_degrees") {
        CHECK_THROWS(batch_norm_get_gamma_weights_parallel_dim_degrees(
            attrs_affine_true, input));
      }

      SUBCASE("batch_norm_get_beta_weights_parallel_dim_degrees") {
        CHECK_THROWS(batch_norm_get_beta_weights_parallel_dim_degrees(
            attrs_affine_true, input));
      }
    }

    SUBCASE("sum parallelism") {
      SumDegree sum_degree = SumDegree{2_p};

      ParallelTensorDimDegrees input = ParallelTensorDimDegrees{
          sum_degree,
          DiscardCopyDegree{1_p},
          FFOrdered{1_p, 1_p, 1_p, 1_p},
      };

      SUBCASE("batch_norm_get_output_parallel_dim_degrees") {
        CHECK_THROWS(batch_norm_get_output_parallel_dim_degrees(
            attrs_affine_true, input));
      }

      SUBCASE("batch_norm_get_gamma_weights_parallel_dim_degrees") {
        CHECK_THROWS(batch_norm_get_gamma_weights_parallel_dim_degrees(
            attrs_affine_true, input));
      }

      SUBCASE("batch_norm_get_beta_weights_parallel_dim_degrees") {
        CHECK_THROWS(batch_norm_get_beta_weights_parallel_dim_degrees(
            attrs_affine_true, input));
      }
    }

    SUBCASE("discard copy parallelism") {
      DiscardCopyDegree discard_copy_degree = DiscardCopyDegree{2_p};

      ParallelTensorDimDegrees input = ParallelTensorDimDegrees{
          SumDegree{1_p},
          discard_copy_degree,
          FFOrdered{1_p, 1_p, 1_p, 1_p},
      };

      SUBCASE("batch_norm_get_output_parallel_dim_degrees") {
        CHECK_THROWS(batch_norm_get_output_parallel_dim_degrees(
            attrs_affine_true, input));
      }

      SUBCASE("batch_norm_get_gamma_weights_parallel_dim_degrees") {
        CHECK_THROWS(batch_norm_get_gamma_weights_parallel_dim_degrees(
            attrs_affine_true, input));
      }

      SUBCASE("batch_norm_get_beta_weights_parallel_dim_degrees") {
        CHECK_THROWS(batch_norm_get_beta_weights_parallel_dim_degrees(
            attrs_affine_true, input));
      }
    }
  }

  TEST_CASE("parallel shape inference (BatchNormAttrs)") {
    // since most of the edge cases are already tested in the above test cases
    // (i.e., shape inference and parallel degree inference)
    // here we just do a basic check that they compose

    BatchNormAttrs attrs = BatchNormAttrs{
        /*activation=*/Activation::RELU,
        /*affine=*/true,
        /*eps=*/1.0,
        /*momentum=*/0.1,
    };

    ParallelTensorShape input = ParallelTensorShape{
        ParallelTensorDims{
            FFOrdered<ShardParallelDim>{
                ShardParallelDim{12_p, 1_p},
                ShardParallelDim{14_p, 2_p},
                ShardParallelDim{16_p, 1_p},
                ShardParallelDim{18_p, 1_p},
            },
            ReplicaParallelDimSet{
                SumDegree{1_p},
                DiscardCopyDegree{1_p},
            },
        },
        DataType::FLOAT,
    };

    SUBCASE("batch_norm_get_output_parallel_shape") {
      ParallelTensorShape result =
          batch_norm_get_output_parallel_shape(attrs, input);
      ParallelTensorShape correct = input;

      CHECK(result == correct);
    }

    SUBCASE("batch_norm_get_gamma_weights_parallel_shape") {
      ParallelTensorShape result =
          batch_norm_get_gamma_weights_parallel_shape(attrs, input);
      ParallelTensorShape correct = ParallelTensorShape{
          ParallelTensorDims{
              FFOrdered<ShardParallelDim>{
                  ShardParallelDim{14_p, 2_p},
              },
              ReplicaParallelDimSet{
                  SumDegree{1_p},
                  DiscardCopyDegree{1_p},
              },
          },
          DataType::FLOAT,
      };

      CHECK(result == correct);
    }

    SUBCASE("batch_norm_get_beta_weights_parallel_shape") {
      ParallelTensorShape result =
          batch_norm_get_beta_weights_parallel_shape(attrs, input);
      ParallelTensorShape correct = ParallelTensorShape{
          ParallelTensorDims{
              FFOrdered<ShardParallelDim>{
                  ShardParallelDim{14_p, 2_p},
              },
              ReplicaParallelDimSet{
                  SumDegree{1_p},
                  DiscardCopyDegree{1_p},
              },
          },
          DataType::FLOAT,
      };

      CHECK(result == correct);
    }
  }

  TEST_CASE("batch_norm_get_task_group") {
    auto mk_attrs = [&](bool affine) -> BatchNormAttrs {
      return BatchNormAttrs{
          /*activation=*/std::nullopt,
          /*affine=*/affine,
          /*eps=*/1.0,
          /*momentum=*/0.1,
      };
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

    SUBCASE("data parallelism") {
      ParallelTensorDimDegrees input_degrees = mk_dim_degrees(1, 1, 2, 1, 1);

      SUBCASE("affine = true") {
        BatchNormAttrs attrs = mk_attrs(/*affine=*/true);

        StandardOperatorTaskGroup result =
            batch_norm_get_task_group(attrs, input_degrees);

        SUBCASE("input degrees") {
          ParallelTensorDimDegrees result_input_degrees =
              parallel_tensor_space_for_standard_operator_task_group_and_slot(
                  result, TensorSlotName::INPUT);

          ParallelTensorDimDegrees correct_input_degrees = input_degrees;

          CHECK(result_input_degrees == correct_input_degrees);
        }

        SUBCASE("gamma weight degrees") {
          ParallelTensorDimDegrees result_gamma_weight_degrees =
              parallel_tensor_space_for_standard_operator_task_group_and_slot(
                  result, TensorSlotName::GAMMA);

          ParallelTensorDimDegrees correct_gamma_weight_degrees =
              batch_norm_get_gamma_weights_parallel_dim_degrees(attrs,
                                                                input_degrees);

          CHECK(result_gamma_weight_degrees == correct_gamma_weight_degrees);
        }

        SUBCASE("beta weight degrees") {
          ParallelTensorDimDegrees result_beta_weight_degrees =
              parallel_tensor_space_for_standard_operator_task_group_and_slot(
                  result, TensorSlotName::BETA);

          ParallelTensorDimDegrees correct_beta_weight_degrees =
              batch_norm_get_beta_weights_parallel_dim_degrees(attrs,
                                                               input_degrees);

          CHECK(result_beta_weight_degrees == correct_beta_weight_degrees);
        }

        SUBCASE("output degrees") {
          ParallelTensorDimDegrees result_output_degrees =
              parallel_tensor_space_for_standard_operator_task_group_and_slot(
                  result, TensorSlotName::OUTPUT);

          ParallelTensorDimDegrees correct_output_degrees =
              batch_norm_get_output_parallel_dim_degrees(attrs, input_degrees);

          CHECK(result_output_degrees == correct_output_degrees);
        }
      }

      SUBCASE("affine = false") {
        BatchNormAttrs attrs = mk_attrs(/*affine=*/false);

        StandardOperatorTaskGroup result =
            batch_norm_get_task_group(attrs, input_degrees);

        SUBCASE("input degrees") {
          ParallelTensorDimDegrees result_input_degrees =
              parallel_tensor_space_for_standard_operator_task_group_and_slot(
                  result, TensorSlotName::INPUT);

          ParallelTensorDimDegrees correct_input_degrees = input_degrees;

          CHECK(result_input_degrees == correct_input_degrees);
        }

        SUBCASE("output degrees") {
          ParallelTensorDimDegrees result_output_degrees =
              parallel_tensor_space_for_standard_operator_task_group_and_slot(
                  result, TensorSlotName::OUTPUT);

          ParallelTensorDimDegrees correct_output_degrees =
              batch_norm_get_output_parallel_dim_degrees(attrs, input_degrees);

          CHECK(result_output_degrees == correct_output_degrees);
        }
      }
    }
  }
}
