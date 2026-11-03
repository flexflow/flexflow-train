#include "op-attrs/ops/conv_2d.h"
#include "utils/integer_conversions.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("get_conv2d_incoming_tensor_roles") {
    auto make_attrs = [](bool use_bias) {
      return Conv2DAttrs{/*out_channels=*/4_p,
                         /*kernel_h=*/3_p,
                         /*kernel_w=*/2_p,
                         /*stride_h=*/2_p,
                         /*stride_w=*/2_p,
                         /*padding_h=*/1_n,
                         /*padding_w=*/1_n,
                         /*groups=*/1_p,
                         /*activation=*/std::nullopt,
                         /*use_bias=*/use_bias};
    };

    SUBCASE("with bias") {
      Conv2DAttrs attrs = make_attrs(/*use_bias=*/true);

      std::map<TensorSlotName, IncomingTensorRole> result =
          get_conv2d_incoming_tensor_roles(attrs);
      std::map<TensorSlotName, IncomingTensorRole> correct = {
          {
              TensorSlotName::INPUT,
              IncomingTensorRole::INPUT,
          },
          {
              TensorSlotName::FILTER,
              IncomingTensorRole::WEIGHT,
          },
          {
              TensorSlotName::BIAS,
              IncomingTensorRole::WEIGHT,
          },
      };

      CHECK(result == correct);
    }

    SUBCASE("without bias") {
      Conv2DAttrs attrs = make_attrs(/*use_bias=*/false);

      std::map<TensorSlotName, IncomingTensorRole> result =
          get_conv2d_incoming_tensor_roles(attrs);
      std::map<TensorSlotName, IncomingTensorRole> correct = {
          {
              TensorSlotName::INPUT,
              IncomingTensorRole::INPUT,
          },
          {
              TensorSlotName::FILTER,
              IncomingTensorRole::WEIGHT,
          },
      };

      CHECK(result == correct);
    }
  }

  TEST_CASE("conv2d_get_kernel_shape") {
    TensorShape input = TensorShape{
        TensorDims{FFOrdered{
            7_p,
            4_p,
            11_p,
            15_p,
        }},
        DataType::FLOAT,
    };

    SUBCASE("groups = 1") {
      Conv2DAttrs attrs = Conv2DAttrs{
          /*out_channels=*/13_p,
          /*kernel_h=*/3_p,
          /*kernel_w=*/4_p,
          /*stride_h=*/1_p,
          /*stride_w=*/1_p,
          /*padding_h=*/0_n,
          /*padding_w=*/0_n,
          /*groups=*/1_p,
          /*activation=*/std::nullopt,
          /*use_bias=*/true,
      };

      TensorShape result = conv2d_get_kernel_shape(attrs, input);

      TensorShape correct = TensorShape{
          TensorDims{FFOrdered{
              13_p,
              4_p,
              3_p,
              4_p,
          }},
          DataType::FLOAT,
      };

      CHECK(result == correct);
    }

    SUBCASE("groups > 1") {
      Conv2DAttrs attrs = Conv2DAttrs{
          /*out_channels=*/13_p,
          /*kernel_h=*/3_p,
          /*kernel_w=*/4_p,
          /*stride_h=*/1_p,
          /*stride_w=*/1_p,
          /*padding_h=*/0_n,
          /*padding_w=*/0_n,
          /*groups=*/2_p,
          /*activation=*/std::nullopt,
          /*use_bias=*/true,
      };

      TensorShape result = conv2d_get_kernel_shape(attrs, input);

      TensorShape correct = TensorShape{
          TensorDims{FFOrdered{
              13_p,
              2_p,
              3_p,
              4_p,
          }},
          DataType::FLOAT,
      };

      CHECK(result == correct);
    }
  }

  TEST_CASE("conv2d_get_bias_shape") {
    TensorShape input = TensorShape{
        TensorDims{FFOrdered{
            7_p,
            4_p,
            11_p,
            15_p,
        }},
        DataType::FLOAT,
    };

    Conv2DAttrs attrs = Conv2DAttrs{
        /*out_channels=*/13_p,
        /*kernel_h=*/3_p,
        /*kernel_w=*/4_p,
        /*stride_h=*/1_p,
        /*stride_w=*/1_p,
        /*padding_h=*/0_n,
        /*padding_w=*/0_n,
        /*groups=*/1_p,
        /*activation=*/std::nullopt,
        /*use_bias=*/true,
    };

    TensorShape result = conv2d_get_bias_shape(attrs, input);

    TensorShape correct = TensorShape{
        TensorDims{FFOrdered{
            13_p,
        }},
        DataType::FLOAT,
    };

    CHECK(result == correct);
  }

  TEST_CASE("conv2d_get_output_shape") {
    TensorShape input = TensorShape{
        TensorDims{FFOrdered{
            7_p,
            4_p,
            11_p,
            15_p,
        }},
        DataType::FLOAT,
    };

    SUBCASE("groups = 1") {
      Conv2DAttrs attrs = Conv2DAttrs{
          /*out_channels=*/13_p,
          /*kernel_h=*/3_p,
          /*kernel_w=*/4_p,
          /*stride_h=*/1_p,
          /*stride_w=*/1_p,
          /*padding_h=*/0_n,
          /*padding_w=*/0_n,
          /*groups=*/1_p,
          /*activation=*/std::nullopt,
          /*use_bias=*/true,
      };

      TensorShape result = conv2d_get_output_shape(attrs, input);

      TensorShape correct = TensorShape{
          TensorDims{FFOrdered{
              7_p,
              13_p,
              9_p,
              12_p,
          }},
          DataType::FLOAT,
      };

      CHECK(result == correct);
    }

    SUBCASE("groups > 1 does not affect the output shape") {
      Conv2DAttrs groups_eq_one = Conv2DAttrs{
          /*out_channels=*/13_p,
          /*kernel_h=*/3_p,
          /*kernel_w=*/4_p,
          /*stride_h=*/1_p,
          /*stride_w=*/1_p,
          /*padding_h=*/0_n,
          /*padding_w=*/0_n,
          /*groups=*/2_p,
          /*activation=*/std::nullopt,
          /*use_bias=*/true,
      };

      Conv2DAttrs groups_gt_one = [&]() {
        Conv2DAttrs result = groups_eq_one;
        result.groups = 2_p;
        return result;
      }();

      TensorShape groups_eq_one_result =
          conv2d_get_output_shape(groups_eq_one, input);
      TensorShape groups_gt_one_result =
          conv2d_get_output_shape(groups_gt_one, input);

      CHECK(groups_eq_one_result == groups_gt_one_result);
    }
  }

  TEST_CASE("conv2d_get_kernel_parallel_dim_degrees") {
    SUBCASE("groups = 1") {
      Conv2DAttrs attrs = Conv2DAttrs{
          /*out_channels=*/24_p,
          /*kernel_h=*/3_p,
          /*kernel_w=*/4_p,
          /*stride_h=*/1_p,
          /*stride_w=*/1_p,
          /*padding_h=*/0_n,
          /*padding_w=*/0_n,
          /*groups=*/1_p,
          /*activation=*/std::nullopt,
          /*use_bias=*/true,
      };

      SUBCASE("data parallelism") {
        ParallelTensorDimDegrees input_degrees = ParallelTensorDimDegrees{
            /*sum_degree=*/SumDegree{1_p},
            /*discard_copy_degree=*/DiscardCopyDegree{1_p},
            /*shard_degrees=*/FFOrdered<positive_int>{4_p, 1_p, 1_p, 1_p},
        };

        ParallelTensorDimDegrees result =
            conv2d_get_kernel_parallel_dim_degrees(attrs, input_degrees);

        ParallelTensorDimDegrees correct = ParallelTensorDimDegrees{
            /*sum_degree=*/SumDegree{1_p},
            /*discard_copy_degree=*/DiscardCopyDegree{4_p},
            /*shard_degrees=*/FFOrdered<positive_int>{1_p, 1_p, 1_p, 1_p},
        };

        CHECK(result == correct);
      }

      SUBCASE("activation height parallelism") {
        ParallelTensorDimDegrees input_degrees = ParallelTensorDimDegrees{
            /*sum_degree=*/SumDegree{1_p},
            /*discard_copy_degree=*/DiscardCopyDegree{1_p},
            /*shard_degrees=*/FFOrdered<positive_int>{1_p, 1_p, 2_p, 1_p},
        };

        // TODO: this should eventually be enabled, once we've figured out how
        // to handle the ghost points
        //
        // ParallelTensorDimDegrees result = conv2d_get_kernel_parallel_dim_degrees(attrs, input_degrees);
        //
        // ParallelTensorDimDegrees correct = ParallelTensorDimDegrees{
        //   /*sum_degree=*/SumDegree{1_p},
        //   /*discard_copy_degree=*/DiscardCopyDegree{2_p},
        //   /*shard_degrees=*/FFOrdered<positive_int>{1_p, 1_p, 1_p, 1_p},
        // };
        //
        // CHECK(result == correct);

        CHECK_THROWS(
            conv2d_get_kernel_parallel_dim_degrees(attrs, input_degrees));
      }

      SUBCASE("activation width parallelism") {
        ParallelTensorDimDegrees input_degrees = ParallelTensorDimDegrees{
            /*sum_degree=*/SumDegree{1_p},
            /*discard_copy_degree=*/DiscardCopyDegree{1_p},
            /*shard_degrees=*/FFOrdered<positive_int>{1_p, 1_p, 1_p, 2_p},
        };

        // TODO: this should eventually be enabled, once we've figured out how
        // to handle the ghost points
        //
        // ParallelTensorDimDegrees result = conv2d_get_kernel_parallel_dim_degrees(attrs, input_degrees);
        //
        // ParallelTensorDimDegrees correct = ParallelTensorDimDegrees{
        //   /*sum_degree=*/SumDegree{1_p},
        //   /*discard_copy_degree=*/DiscardCopyDegree{2_p},
        //   /*shard_degrees=*/FFOrdered<positive_int>{1_p, 1_p, 1_p, 1_p},
        // };
        //
        // CHECK(result == correct);

        CHECK_THROWS(
            conv2d_get_kernel_parallel_dim_degrees(attrs, input_degrees));
      }

      SUBCASE("input channel parallelism") {
        ParallelTensorDimDegrees input_degrees = ParallelTensorDimDegrees{
            /*sum_degree=*/SumDegree{1_p},
            /*discard_copy_degree=*/DiscardCopyDegree{1_p},
            /*shard_degrees=*/FFOrdered<positive_int>{1_p, 4_p, 1_p, 1_p},
        };

        ParallelTensorDimDegrees result =
            conv2d_get_kernel_parallel_dim_degrees(attrs, input_degrees);

        ParallelTensorDimDegrees correct = ParallelTensorDimDegrees{
            /*sum_degree=*/SumDegree{1_p},
            /*discard_copy_degree=*/DiscardCopyDegree{1_p},
            /*shard_degrees=*/FFOrdered<positive_int>{1_p, 4_p, 1_p, 1_p},
        };

        CHECK(result == correct);
      }

      SUBCASE("output channel parallelism") {
        ParallelTensorDimDegrees input_degrees = ParallelTensorDimDegrees{
            /*sum_degree=*/SumDegree{1_p},
            /*discard_copy_degree=*/DiscardCopyDegree{4_p},
            /*shard_degrees=*/FFOrdered<positive_int>{1_p, 1_p, 1_p, 1_p},
        };

        ParallelTensorDimDegrees result =
            conv2d_get_kernel_parallel_dim_degrees(attrs, input_degrees);

        ParallelTensorDimDegrees correct = ParallelTensorDimDegrees{
            /*sum_degree=*/SumDegree{1_p},
            /*discard_copy_degree=*/DiscardCopyDegree{1_p},
            /*shard_degrees=*/FFOrdered<positive_int>{4_p, 1_p, 1_p, 1_p},
        };

        CHECK(result == correct);
      }

      SUBCASE("propagating sum degree") {
        ParallelTensorDimDegrees input_degrees = ParallelTensorDimDegrees{
            /*sum_degree=*/SumDegree{4_p},
            /*discard_copy_degree=*/DiscardCopyDegree{1_p},
            /*shard_degrees=*/FFOrdered<positive_int>{1_p, 1_p, 1_p, 1_p},
        };

        ParallelTensorDimDegrees result =
            conv2d_get_kernel_parallel_dim_degrees(attrs, input_degrees);

        ParallelTensorDimDegrees correct = ParallelTensorDimDegrees{
            /*sum_degree=*/SumDegree{1_p},
            /*discard_copy_degree=*/DiscardCopyDegree{4_p},
            /*shard_degrees=*/FFOrdered<positive_int>{1_p, 1_p, 1_p, 1_p},
        };

        CHECK(result == correct);
      }
    }

    SUBCASE("groups > 1") {
      Conv2DAttrs attrs = Conv2DAttrs{
          /*out_channels=*/24_p,
          /*kernel_h=*/3_p,
          /*kernel_w=*/3_p,
          /*stride_h=*/1_p,
          /*stride_w=*/1_p,
          /*padding_h=*/0_n,
          /*padding_w=*/0_n,
          /*groups=*/4_p,
          /*activation=*/std::nullopt,
          /*use_bias=*/true,
      };

      SUBCASE("groups == input channel degree") {
        ParallelTensorDimDegrees input_degrees = ParallelTensorDimDegrees{
            /*sum_degree=*/SumDegree{1_p},
            /*discard_copy_degree=*/DiscardCopyDegree{1_p},
            /*shard_degrees=*/FFOrdered<positive_int>{1_p, 4_p, 1_p, 1_p},
        };

        ParallelTensorDimDegrees result =
            conv2d_get_kernel_parallel_dim_degrees(attrs, input_degrees);

        ParallelTensorDimDegrees correct = ParallelTensorDimDegrees{
            /*sum_degree=*/SumDegree{1_p},
            /*discard_copy_degree=*/DiscardCopyDegree{1_p},
            /*shard_degrees=*/FFOrdered<positive_int>{4_p, 1_p, 1_p, 1_p},
        };

        CHECK(result == correct);
      }

      SUBCASE("groups is divisible by input channel degree") {
        ParallelTensorDimDegrees input_degrees = ParallelTensorDimDegrees{
            /*sum_degree=*/SumDegree{1_p},
            /*discard_copy_degree=*/DiscardCopyDegree{1_p},
            /*shard_degrees=*/FFOrdered<positive_int>{1_p, 2_p, 1_p, 1_p},
        };

        ParallelTensorDimDegrees result =
            conv2d_get_kernel_parallel_dim_degrees(attrs, input_degrees);

        ParallelTensorDimDegrees correct = ParallelTensorDimDegrees{
            /*sum_degree=*/SumDegree{1_p},
            /*discard_copy_degree=*/DiscardCopyDegree{1_p},
            /*shard_degrees=*/FFOrdered<positive_int>{2_p, 1_p, 1_p, 1_p},
        };

        CHECK(result == correct);
      }

      SUBCASE("input channel degree is divisible by groups") {
        ParallelTensorDimDegrees input_degrees = ParallelTensorDimDegrees{
            /*sum_degree=*/SumDegree{1_p},
            /*discard_copy_degree=*/DiscardCopyDegree{1_p},
            /*shard_degrees=*/FFOrdered<positive_int>{1_p, 24_p, 1_p, 1_p},
        };

        ParallelTensorDimDegrees result =
            conv2d_get_kernel_parallel_dim_degrees(attrs, input_degrees);

        ParallelTensorDimDegrees correct = ParallelTensorDimDegrees{
            /*sum_degree=*/SumDegree{1_p},
            /*discard_copy_degree=*/DiscardCopyDegree{1_p},
            /*shard_degrees=*/FFOrdered<positive_int>{4_p, 6_p, 1_p, 1_p},
        };

        CHECK(result == correct);
      }

      SUBCASE("input channel degree and groups are incompatible") {
        ParallelTensorDimDegrees input_degrees = ParallelTensorDimDegrees{
            /*sum_degree=*/SumDegree{1_p},
            /*discard_copy_degree=*/DiscardCopyDegree{1_p},
            /*shard_degrees=*/FFOrdered<positive_int>{1_p, 6_p, 1_p, 1_p},
        };

        CHECK_THROWS(
            conv2d_get_kernel_parallel_dim_degrees(attrs, input_degrees));
      }
    }
  }

  TEST_CASE("conv2d_get_bias_parallel_dim_degrees") {
    SUBCASE("groups = 1") {
      Conv2DAttrs attrs = Conv2DAttrs{
          /*out_channels=*/24_p,
          /*kernel_h=*/3_p,
          /*kernel_w=*/4_p,
          /*stride_h=*/1_p,
          /*stride_w=*/1_p,
          /*padding_h=*/0_n,
          /*padding_w=*/0_n,
          /*groups=*/1_p,
          /*activation=*/std::nullopt,
          /*use_bias=*/true,
      };

      SUBCASE("data parallelism") {
        ParallelTensorDimDegrees input_degrees = ParallelTensorDimDegrees{
            /*sum_degree=*/SumDegree{1_p},
            /*discard_copy_degree=*/DiscardCopyDegree{1_p},
            /*shard_degrees=*/FFOrdered<positive_int>{4_p, 1_p, 1_p, 1_p},
        };

        ParallelTensorDimDegrees result =
            conv2d_get_bias_parallel_dim_degrees(attrs, input_degrees);

        ParallelTensorDimDegrees correct = ParallelTensorDimDegrees{
            /*sum_degree=*/SumDegree{1_p},
            /*discard_copy_degree=*/DiscardCopyDegree{4_p},
            /*shard_degrees=*/FFOrdered<positive_int>{1_p},
        };

        CHECK(result == correct);
      }

      SUBCASE("activation height parallelism") {
        ParallelTensorDimDegrees input_degrees = ParallelTensorDimDegrees{
            /*sum_degree=*/SumDegree{1_p},
            /*discard_copy_degree=*/DiscardCopyDegree{1_p},
            /*shard_degrees=*/FFOrdered<positive_int>{1_p, 1_p, 2_p, 1_p},
        };

        // TODO: this should eventually be enabled, once we've figured out how
        // to handle the ghost points
        //
        // ParallelTensorDimDegrees result = conv2d_get_bias_parallel_dim_degrees(attrs, input_degrees);
        //
        // ParallelTensorDimDegrees correct = ParallelTensorDimDegrees{
        //   /*sum_degree=*/SumDegree{2_p},
        //   /*discard_copy_degree=*/DiscardCopyDegree{1_p},
        //   /*shard_degrees=*/FFOrdered<positive_int>{1_p},
        // };
        //
        // CHECK(result == correct);

        CHECK_THROWS(
            conv2d_get_bias_parallel_dim_degrees(attrs, input_degrees));
      }

      SUBCASE("activation width parallelism") {
        ParallelTensorDimDegrees input_degrees = ParallelTensorDimDegrees{
            /*sum_degree=*/SumDegree{1_p},
            /*discard_copy_degree=*/DiscardCopyDegree{1_p},
            /*shard_degrees=*/FFOrdered<positive_int>{1_p, 1_p, 1_p, 2_p},
        };

        // TODO: this should eventually be enabled, once we've figured out how
        // to handle the ghost points
        //
        // ParallelTensorDimDegrees result = conv2d_get_bias_parallel_dim_degrees(attrs, input_degrees);
        //
        // ParallelTensorDimDegrees correct = ParallelTensorDimDegrees{
        //   /*sum_degree=*/SumDegree{2_p},
        //   /*discard_copy_degree=*/DiscardCopyDegree{1_p},
        //   /*shard_degrees=*/FFOrdered<positive_int>{1_p},
        // };
        //
        // CHECK(result == correct);

        CHECK_THROWS(
            conv2d_get_bias_parallel_dim_degrees(attrs, input_degrees));
      }

      SUBCASE("input channel parallelism") {
        ParallelTensorDimDegrees input_degrees = ParallelTensorDimDegrees{
            /*sum_degree=*/SumDegree{1_p},
            /*discard_copy_degree=*/DiscardCopyDegree{1_p},
            /*shard_degrees=*/FFOrdered<positive_int>{1_p, 4_p, 1_p, 1_p},
        };

        ParallelTensorDimDegrees result =
            conv2d_get_bias_parallel_dim_degrees(attrs, input_degrees);

        ParallelTensorDimDegrees correct = ParallelTensorDimDegrees{
            /*sum_degree=*/SumDegree{4_p},
            /*discard_copy_degree=*/DiscardCopyDegree{1_p},
            /*shard_degrees=*/FFOrdered<positive_int>{1_p},
        };

        CHECK(result == correct);
      }

      SUBCASE("output channel parallelism") {
        ParallelTensorDimDegrees input_degrees = ParallelTensorDimDegrees{
            /*sum_degree=*/SumDegree{1_p},
            /*discard_copy_degree=*/DiscardCopyDegree{4_p},
            /*shard_degrees=*/FFOrdered<positive_int>{1_p, 1_p, 1_p, 1_p},
        };

        ParallelTensorDimDegrees result =
            conv2d_get_bias_parallel_dim_degrees(attrs, input_degrees);

        ParallelTensorDimDegrees correct = ParallelTensorDimDegrees{
            /*sum_degree=*/SumDegree{1_p},
            /*discard_copy_degree=*/DiscardCopyDegree{1_p},
            /*shard_degrees=*/FFOrdered<positive_int>{4_p},
        };

        CHECK(result == correct);
      }

      SUBCASE("propagating sum degree") {
        ParallelTensorDimDegrees input_degrees = ParallelTensorDimDegrees{
            /*sum_degree=*/SumDegree{4_p},
            /*discard_copy_degree=*/DiscardCopyDegree{1_p},
            /*shard_degrees=*/FFOrdered<positive_int>{1_p, 1_p, 1_p, 1_p},
        };

        ParallelTensorDimDegrees result =
            conv2d_get_bias_parallel_dim_degrees(attrs, input_degrees);

        ParallelTensorDimDegrees correct = ParallelTensorDimDegrees{
            /*sum_degree=*/SumDegree{4_p},
            /*discard_copy_degree=*/DiscardCopyDegree{1_p},
            /*shard_degrees=*/FFOrdered<positive_int>{1_p},
        };

        CHECK(result == correct);
      }
    }

    SUBCASE("groups > 1") {
      Conv2DAttrs attrs = Conv2DAttrs{
          /*out_channels=*/24_p,
          /*kernel_h=*/3_p,
          /*kernel_w=*/3_p,
          /*stride_h=*/1_p,
          /*stride_w=*/1_p,
          /*padding_h=*/0_n,
          /*padding_w=*/0_n,
          /*groups=*/4_p,
          /*activation=*/std::nullopt,
          /*use_bias=*/true,
      };

      SUBCASE("groups == input channel degree") {
        ParallelTensorDimDegrees input_degrees = ParallelTensorDimDegrees{
            /*sum_degree=*/SumDegree{1_p},
            /*discard_copy_degree=*/DiscardCopyDegree{1_p},
            /*shard_degrees=*/FFOrdered<positive_int>{1_p, 4_p, 1_p, 1_p},
        };

        ParallelTensorDimDegrees result =
            conv2d_get_bias_parallel_dim_degrees(attrs, input_degrees);

        ParallelTensorDimDegrees correct = ParallelTensorDimDegrees{
            /*sum_degree=*/SumDegree{1_p},
            /*discard_copy_degree=*/DiscardCopyDegree{1_p},
            /*shard_degrees=*/FFOrdered<positive_int>{4_p},
        };

        CHECK(result == correct);
      }

      SUBCASE("groups is divisible by input channel degree") {
        ParallelTensorDimDegrees input_degrees = ParallelTensorDimDegrees{
            /*sum_degree=*/SumDegree{1_p},
            /*discard_copy_degree=*/DiscardCopyDegree{1_p},
            /*shard_degrees=*/FFOrdered<positive_int>{1_p, 2_p, 1_p, 1_p},
        };

        ParallelTensorDimDegrees result =
            conv2d_get_bias_parallel_dim_degrees(attrs, input_degrees);

        ParallelTensorDimDegrees correct = ParallelTensorDimDegrees{
            /*sum_degree=*/SumDegree{1_p},
            /*discard_copy_degree=*/DiscardCopyDegree{1_p},
            /*shard_degrees=*/FFOrdered<positive_int>{2_p},
        };

        CHECK(result == correct);
      }

      SUBCASE("input channel degree is divisible by groups") {
        ParallelTensorDimDegrees input_degrees = ParallelTensorDimDegrees{
            /*sum_degree=*/SumDegree{1_p},
            /*discard_copy_degree=*/DiscardCopyDegree{1_p},
            /*shard_degrees=*/FFOrdered<positive_int>{1_p, 24_p, 1_p, 1_p},
        };

        ParallelTensorDimDegrees result =
            conv2d_get_bias_parallel_dim_degrees(attrs, input_degrees);

        ParallelTensorDimDegrees correct = ParallelTensorDimDegrees{
            /*sum_degree=*/SumDegree{6_p},
            /*discard_copy_degree=*/DiscardCopyDegree{1_p},
            /*shard_degrees=*/FFOrdered<positive_int>{4_p},
        };

        CHECK(result == correct);
      }

      SUBCASE("input channel degree and groups are incompatible") {
        ParallelTensorDimDegrees input_degrees = ParallelTensorDimDegrees{
            /*sum_degree=*/SumDegree{1_p},
            /*discard_copy_degree=*/DiscardCopyDegree{1_p},
            /*shard_degrees=*/FFOrdered<positive_int>{1_p, 6_p, 1_p, 1_p},
        };

        CHECK_THROWS(
            conv2d_get_bias_parallel_dim_degrees(attrs, input_degrees));
      }
    }
  }

  TEST_CASE("conv2d_get_output_parallel_dim_degrees") {
    SUBCASE("groups = 1") {
      Conv2DAttrs attrs = Conv2DAttrs{
          /*out_channels=*/24_p,
          /*kernel_h=*/3_p,
          /*kernel_w=*/3_p,
          /*stride_h=*/1_p,
          /*stride_w=*/1_p,
          /*padding_h=*/0_n,
          /*padding_w=*/0_n,
          /*groups=*/1_p,
          /*activation=*/std::nullopt,
          /*use_bias=*/true,
      };

      SUBCASE("data parallelism") {
        ParallelTensorDimDegrees input_degrees = ParallelTensorDimDegrees{
            /*sum_degree=*/SumDegree{1_p},
            /*discard_copy_degree=*/DiscardCopyDegree{1_p},
            /*shard_degrees=*/FFOrdered<positive_int>{4_p, 1_p, 1_p, 1_p},
        };

        ParallelTensorDimDegrees result =
            conv2d_get_output_parallel_dim_degrees(attrs, input_degrees);

        ParallelTensorDimDegrees correct = ParallelTensorDimDegrees{
            /*sum_degree=*/SumDegree{1_p},
            /*discard_copy_degree=*/DiscardCopyDegree{1_p},
            /*shard_degrees=*/FFOrdered<positive_int>{4_p, 1_p, 1_p, 1_p},
        };

        CHECK(result == correct);
      }

      SUBCASE("activation height parallelism") {
        ParallelTensorDimDegrees input_degrees = ParallelTensorDimDegrees{
            /*sum_degree=*/SumDegree{1_p},
            /*discard_copy_degree=*/DiscardCopyDegree{1_p},
            /*shard_degrees=*/FFOrdered<positive_int>{1_p, 1_p, 2_p, 1_p},
        };

        // TODO: this should eventually be enabled, once we've figured out how
        // to handle the ghost points
        //
        // ParallelTensorDimDegrees result = conv2d_get_output_parallel_dim_degrees(attrs, input_degrees);
        //
        // ParallelTensorDimDegrees correct = ParallelTensorDimDegrees{
        //   /*sum_degree=*/SumDegree{1_p},
        //   /*discard_copy_degree=*/DiscardCopyDegree{1_p},
        //   /*shard_degrees=*/FFOrdered<positive_int>{1_p, 1_p, 2_p, 1_p},
        // };
        //
        // CHECK(result == correct);

        CHECK_THROWS(
            conv2d_get_output_parallel_dim_degrees(attrs, input_degrees));
      }

      SUBCASE("activation width parallelism") {
        ParallelTensorDimDegrees input_degrees = ParallelTensorDimDegrees{
            /*sum_degree=*/SumDegree{1_p},
            /*discard_copy_degree=*/DiscardCopyDegree{1_p},
            /*shard_degrees=*/FFOrdered<positive_int>{1_p, 1_p, 1_p, 2_p},
        };

        // TODO: this should eventually be enabled, once we've figured out how
        // to handle the ghost points
        //
        // ParallelTensorDimDegrees result = conv2d_get_output_parallel_dim_degrees(attrs, input_degrees);
        //
        // ParallelTensorDimDegrees correct = ParallelTensorDimDegrees{
        //   /*sum_degree=*/SumDegree{1_p},
        //   /*discard_copy_degree=*/DiscardCopyDegree{1_p},
        //   /*shard_degrees=*/FFOrdered<positive_int>{1_p, 1_p, 1_p, 2_p},
        // };
        //
        // CHECK(result == correct);

        CHECK_THROWS(
            conv2d_get_output_parallel_dim_degrees(attrs, input_degrees));
      }

      SUBCASE("input channel parallelism") {
        ParallelTensorDimDegrees input_degrees = ParallelTensorDimDegrees{
            /*sum_degree=*/SumDegree{1_p},
            /*discard_copy_degree=*/DiscardCopyDegree{1_p},
            /*shard_degrees=*/FFOrdered<positive_int>{1_p, 4_p, 1_p, 1_p},
        };

        ParallelTensorDimDegrees result =
            conv2d_get_output_parallel_dim_degrees(attrs, input_degrees);

        ParallelTensorDimDegrees correct = ParallelTensorDimDegrees{
            /*sum_degree=*/SumDegree{4_p},
            /*discard_copy_degree=*/DiscardCopyDegree{1_p},
            /*shard_degrees=*/FFOrdered<positive_int>{1_p, 1_p, 1_p, 1_p},
        };

        CHECK(result == correct);
      }

      SUBCASE("output channel parallelism") {
        ParallelTensorDimDegrees input_degrees = ParallelTensorDimDegrees{
            /*sum_degree=*/SumDegree{1_p},
            /*discard_copy_degree=*/DiscardCopyDegree{4_p},
            /*shard_degrees=*/FFOrdered<positive_int>{1_p, 1_p, 1_p, 1_p},
        };

        ParallelTensorDimDegrees result =
            conv2d_get_output_parallel_dim_degrees(attrs, input_degrees);

        ParallelTensorDimDegrees correct = ParallelTensorDimDegrees{
            /*sum_degree=*/SumDegree{1_p},
            /*discard_copy_degree=*/DiscardCopyDegree{1_p},
            /*shard_degrees=*/FFOrdered<positive_int>{1_p, 4_p, 1_p, 1_p},
        };

        CHECK(result == correct);
      }

      SUBCASE("propagating sum degree") {
        ParallelTensorDimDegrees input_degrees = ParallelTensorDimDegrees{
            /*sum_degree=*/SumDegree{4_p},
            /*discard_copy_degree=*/DiscardCopyDegree{1_p},
            /*shard_degrees=*/FFOrdered<positive_int>{1_p, 1_p, 1_p, 1_p},
        };

        ParallelTensorDimDegrees result =
            conv2d_get_output_parallel_dim_degrees(attrs, input_degrees);

        ParallelTensorDimDegrees correct = ParallelTensorDimDegrees{
            /*sum_degree=*/SumDegree{4_p},
            /*discard_copy_degree=*/DiscardCopyDegree{1_p},
            /*shard_degrees=*/FFOrdered<positive_int>{1_p, 1_p, 1_p, 1_p},
        };

        CHECK(result == correct);
      }
    }

    SUBCASE("groups > 1") {
      Conv2DAttrs attrs = Conv2DAttrs{
          /*out_channels=*/24_p,
          /*kernel_h=*/3_p,
          /*kernel_w=*/3_p,
          /*stride_h=*/1_p,
          /*stride_w=*/1_p,
          /*padding_h=*/0_n,
          /*padding_w=*/0_n,
          /*groups=*/4_p,
          /*activation=*/std::nullopt,
          /*use_bias=*/true,
      };

      SUBCASE("groups == input channel degree") {
        ParallelTensorDimDegrees input_degrees = ParallelTensorDimDegrees{
            /*sum_degree=*/SumDegree{1_p},
            /*discard_copy_degree=*/DiscardCopyDegree{1_p},
            /*shard_degrees=*/FFOrdered<positive_int>{1_p, 4_p, 1_p, 1_p},
        };

        ParallelTensorDimDegrees result =
            conv2d_get_output_parallel_dim_degrees(attrs, input_degrees);

        ParallelTensorDimDegrees correct = ParallelTensorDimDegrees{
            /*sum_degree=*/SumDegree{1_p},
            /*discard_copy_degree=*/DiscardCopyDegree{1_p},
            /*shard_degrees=*/FFOrdered<positive_int>{1_p, 4_p, 1_p, 1_p},
        };

        CHECK(result == correct);
      }

      SUBCASE("groups is divisible by input channel degree") {
        ParallelTensorDimDegrees input_degrees = ParallelTensorDimDegrees{
            /*sum_degree=*/SumDegree{1_p},
            /*discard_copy_degree=*/DiscardCopyDegree{1_p},
            /*shard_degrees=*/FFOrdered<positive_int>{1_p, 2_p, 1_p, 1_p},
        };

        ParallelTensorDimDegrees result =
            conv2d_get_output_parallel_dim_degrees(attrs, input_degrees);

        ParallelTensorDimDegrees correct = ParallelTensorDimDegrees{
            /*sum_degree=*/SumDegree{1_p},
            /*discard_copy_degree=*/DiscardCopyDegree{1_p},
            /*shard_degrees=*/FFOrdered<positive_int>{1_p, 2_p, 1_p, 1_p},
        };

        CHECK(result == correct);
      }

      SUBCASE("input channel degree is divisible by groups") {
        ParallelTensorDimDegrees input_degrees = ParallelTensorDimDegrees{
            /*sum_degree=*/SumDegree{1_p},
            /*discard_copy_degree=*/DiscardCopyDegree{1_p},
            /*shard_degrees=*/FFOrdered<positive_int>{1_p, 24_p, 1_p, 1_p},
        };

        ParallelTensorDimDegrees result =
            conv2d_get_output_parallel_dim_degrees(attrs, input_degrees);

        ParallelTensorDimDegrees correct = ParallelTensorDimDegrees{
            /*sum_degree=*/SumDegree{6_p},
            /*discard_copy_degree=*/DiscardCopyDegree{1_p},
            /*shard_degrees=*/FFOrdered<positive_int>{1_p, 4_p, 1_p, 1_p},
        };

        CHECK(result == correct);
      }

      SUBCASE("input channel degree and groups are incompatible") {
        ParallelTensorDimDegrees input_degrees = ParallelTensorDimDegrees{
            /*sum_degree=*/SumDegree{1_p},
            /*discard_copy_degree=*/DiscardCopyDegree{1_p},
            /*shard_degrees=*/FFOrdered<positive_int>{1_p, 6_p, 1_p, 1_p},
        };

        CHECK_THROWS(
            conv2d_get_output_parallel_dim_degrees(attrs, input_degrees));
      }
    }
  }
}
