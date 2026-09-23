#include "models/yolov10/yolov10.h"
#include "pcg/computation_graph.h"
#include "utils/containers/are_all_distinct.h"
#include "utils/containers/filtrans.h"
#include "utils/containers/foldl.h"
#include "utils/containers/generate_map.h"
#include "utils/containers/get_only.h"
#include "utils/containers/map_values.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("create_yolov10_v10detect_cls_head") {
    ComputationGraphBuilder b;

    TensorShape input_shape = TensorShape{
        TensorDims{
            FFOrdered<positive_int>{
                1_p,
                512_p,
                40_p,
                40_p,
            },
        },
        DataType::FLOAT,
    };

    tensor_guid_t input_tensor = b.create_input(input_shape, CreateGrad::NO);

    tensor_guid_t result = create_yolov10_v10detect_cls_head(
        /*cgb=*/b,
        /*input_tensor=*/input_tensor,
        /*c3=*/31_p,
        /*num_classes=*/76_p);

    SUBCASE("produces correct output shapes") {
      TensorShape result_shape = b.get_shape(result);

      TensorShape correct_shape = TensorShape{
          TensorDims{
              FFOrdered<positive_int>{
                  1_p,
                  76_p,
                  40_p,
                  40_p,
              },
          },
          DataType::FLOAT,
      };

      CHECK(result_shape == correct_shape);
    }

    SUBCASE("contains expected numbers of operators") {
      std::map<OperatorType, positive_int> result_op_type_counts =
          operator_type_counts_in_computation_graph(b.computation_graph);

      std::map<OperatorType, positive_int> correct_op_type_counts = {
          {OperatorType::INPUT, 1_p},
          {OperatorType::CONV2D, 5_p},
          {OperatorType::BATCHNORM, 4_p},
          {OperatorType::WEIGHT, 14_p},
          {OperatorType::SILU, 4_p},
      };

      CHECK(result_op_type_counts == correct_op_type_counts);
    }
  }

  TEST_CASE("create_yolov10_v10detect_box_head") {
    ComputationGraphBuilder b;

    TensorShape input_shape = TensorShape{
        TensorDims{
            FFOrdered<positive_int>{
                1_p,
                512_p,
                40_p,
                40_p,
            },
        },
        DataType::FLOAT,
    };

    tensor_guid_t input_tensor = b.create_input(input_shape, CreateGrad::NO);

    tensor_guid_t result = create_yolov10_v10detect_box_head(
        /*cgb=*/b,
        /*input_tensor=*/input_tensor,
        /*c2=*/31_p,
        /*reg_max=*/18_p);

    SUBCASE("produces correct output shapes") {
      TensorShape result_shape = b.get_shape(result);

      TensorShape correct_shape = TensorShape{
          TensorDims{
              FFOrdered<positive_int>{
                  1_p,
                  4_p * 18_p,
                  40_p,
                  40_p,
              },
          },
          DataType::FLOAT,
      };

      CHECK(result_shape == correct_shape);
    }

    SUBCASE("contains expected numbers of operators") {
      std::map<OperatorType, positive_int> result_op_type_counts =
          operator_type_counts_in_computation_graph(b.computation_graph);

      std::map<OperatorType, positive_int> correct_op_type_counts = {
          {OperatorType::INPUT, 1_p},
          {OperatorType::CONV2D, 3_p},
          {OperatorType::BATCHNORM, 2_p},
          {OperatorType::WEIGHT, 8_p},
          {OperatorType::SILU, 2_p},
      };

      CHECK(result_op_type_counts == correct_op_type_counts);
    }
  }

  TEST_CASE("create_yolov10_v10detect_module") {
    /* example tensor shapes pulled from
     * https://github.com/ultralytics/ultralytics/blob/f8ad132a15b5f6818c2ce0647b40dc57e993bf0c/ultralytics/nn/modules/head.py#L72-L75
     */

    ComputationGraphBuilder b;

    auto mk_input = [&](positive_int d1,
                        positive_int d2,
                        positive_int d3,
                        positive_int d4) -> tensor_guid_t {
      TensorShape input_shape = TensorShape{
          TensorDims{
              FFOrdered<positive_int>{
                  d1,
                  d2,
                  d3,
                  d4,
              },
          },
          DataType::FLOAT,
      };

      return b.create_input(input_shape, CreateGrad::NO);
    };

    std::vector<tensor_guid_t> inputs = {
        mk_input(1_p, 256_p, 80_p, 80_p),
        mk_input(1_p, 512_p, 40_p, 40_p),
        mk_input(1_p, 1024_p, 20_p, 20_p),
    };

    YOLOv10DetectHeadOutputs result = create_yolov10_v10detect_module(
        /*cgb=*/b,
        /*feats=*/inputs,
        /*num_classes=*/80_p);

    SUBCASE("produces correct output shapes") {
      SUBCASE("boxes tensor") {
        TensorShape result_shape = b.get_shape(result.boxes);

        TensorShape correct_shape = TensorShape{
            TensorDims{
                FFOrdered<positive_int>{
                    1_p,
                    4_p * 16_p,
                    80_p * 80_p + 40_p * 40_p + 20_p * 20_p,
                },
            },
            DataType::FLOAT,
        };

        CHECK(result_shape == correct_shape);
      }

      SUBCASE("scores tensor") {
        TensorShape result_shape = b.get_shape(result.scores);

        TensorShape correct_shape = TensorShape{
            TensorDims{
                FFOrdered<positive_int>{
                    1_p,
                    80_p,
                    80_p * 80_p + 40_p * 40_p + 20_p * 20_p,
                },
            },
            DataType::FLOAT,
        };

        CHECK(result_shape == correct_shape);
      }
    }

    SUBCASE("contains expected numbers of operators") {
      std::map<OperatorType, positive_int> result_op_type_counts =
          operator_type_counts_in_computation_graph(b.computation_graph);

      std::map<OperatorType, positive_int> correct_op_type_counts = {
          {OperatorType::INPUT, 3_p},
          {OperatorType::CONV2D, 24_p},
          {OperatorType::BATCHNORM, 18_p},
          {OperatorType::WEIGHT, 66_p},
          {OperatorType::SILU, 18_p},
          {OperatorType::RESHAPE, 6_p},
          {OperatorType::CONCAT, 2_p},
      };

      CHECK(result_op_type_counts == correct_op_type_counts);
    }
  }

  TEST_CASE("create_yolov10_conv_module") {
    ComputationGraphBuilder b;

    TensorShape input_shape = TensorShape{
        TensorDims{
            FFOrdered<positive_int>{
                1_p,
                64_p,
                13_p,
                8_p,
            },
        },
        DataType::FLOAT,
    };

    tensor_guid_t input_tensor = b.create_input(input_shape, CreateGrad::NO);

    SUBCASE("for stride = 1 and num_input_channels == num_output_channels, "
            "produces same sized output") {
      tensor_guid_t result = create_yolov10_conv_module(
          /*cgb=*/b,
          /*input_tensor=*/input_tensor,
          /*num_input_channels=*/64_p,
          /*num_output_channels=*/64_p,
          /*kernel_size=*/9_p,
          /*stride=*/1_p,
          /*groups=*/1_p,
          /*use_activation=*/true);

      TensorShape result_output_shape = b.get_shape(result);

      TensorShape correct_output_shape = input_shape;

      CHECK(result_output_shape == correct_output_shape);
    }

    SUBCASE("stride != 1") {
      tensor_guid_t result = create_yolov10_conv_module(
          /*cgb=*/b,
          /*input_tensor=*/input_tensor,
          /*num_input_channels=*/64_p,
          /*num_output_channels=*/128_p,
          /*kernel_size=*/3_p,
          /*stride=*/2_p,
          /*groups=*/1_p,
          /*use_activation=*/true);

      SUBCASE("produces correct output shape") {
        TensorShape result_output_shape = b.get_shape(result);

        TensorShape correct_output_shape = TensorShape{
            TensorDims{
                FFOrdered<positive_int>{
                    1_p,
                    128_p,
                    7_p,
                    4_p,
                },
            },
            DataType::FLOAT,
        };

        CHECK(result_output_shape == correct_output_shape);
      }

      SUBCASE("contains expected number of operators") {
        std::map<OperatorType, positive_int> result_op_type_counts =
            operator_type_counts_in_computation_graph(b.computation_graph);

        std::map<OperatorType, positive_int> correct_op_type_counts = {
            {OperatorType::INPUT, 1_p},
            {OperatorType::CONV2D, 1_p},
            {OperatorType::BATCHNORM, 1_p},
            {OperatorType::SILU, 1_p},
            {OperatorType::WEIGHT, 3_p},
        };

        CHECK(result_op_type_counts == correct_op_type_counts);
      }
    }

    SUBCASE("large stride") {
      tensor_guid_t result = create_yolov10_conv_module(
          /*cgb=*/b,
          /*input_tensor=*/input_tensor,
          /*num_input_channels=*/64_p,
          /*num_output_channels=*/128_p,
          /*kernel_size=*/3_p,
          /*stride=*/6_p,
          /*groups=*/1_p,
          /*use_activation=*/true);

      SUBCASE("produces correct output shape") {
        TensorShape result_output_shape = b.get_shape(result);

        TensorShape correct_output_shape = TensorShape{
            TensorDims{
                FFOrdered<positive_int>{
                    1_p,
                    128_p,
                    3_p,
                    2_p,
                },
            },
            DataType::FLOAT,
        };

        CHECK(result_output_shape == correct_output_shape);
      }
    }
  }

  TEST_CASE("create_yolov10_scdown_module") {
    // example tensor shapes pulled from
    // https://github.com/ultralytics/ultralytics/blob/f8ad132a15b5f6818c2ce0647b40dc57e993bf0c/ultralytics/nn/modules/block.py#L1543-L1550

    ComputationGraphBuilder b;
    tensor_guid_t input_tensor = b.create_input(
        TensorShape{
            TensorDims{
                FFOrdered<positive_int>{
                    1_p,
                    64_p,
                    128_p,
                    128_p,
                },
            },
            DataType::FLOAT,
        },
        CreateGrad::NO);

    tensor_guid_t result = create_yolov10_scdown_module(
        /*cgb=*/b,
        /*input_tensor=*/input_tensor,
        /*num_input_channels=*/64_p,
        /*num_output_channels=*/128_p,
        /*kernel_size=*/3_p,
        /*stride=*/2_p);

    SUBCASE("produces correct output shape") {
      TensorShape result_output_shape = b.get_shape(result);

      TensorShape correct_output_shape = TensorShape{
          TensorDims{
              FFOrdered<positive_int>{
                  1_p,
                  128_p,
                  64_p,
                  64_p,
              },
          },
          DataType::FLOAT,
      };

      CHECK(result_output_shape == correct_output_shape);
    }

    SUBCASE("contains expected number of operators") {
      std::map<OperatorType, positive_int> result_op_type_counts =
          operator_type_counts_in_computation_graph(b.computation_graph);

      std::map<OperatorType, positive_int> correct_op_type_counts = {
          {OperatorType::INPUT, 1_p},
          {OperatorType::CONV2D, 2_p},
          {OperatorType::BATCHNORM, 2_p},
          {OperatorType::SILU, 1_p},
          {OperatorType::WEIGHT, 6_p},
      };

      CHECK(result_op_type_counts == correct_op_type_counts);
    }
  }

  TEST_CASE("create_yolov10_c2f_module") {
    ComputationGraphBuilder b;
    TensorShape input_shape = TensorShape{
        TensorDims{
            FFOrdered<positive_int>{
                2_p,
                128_p,
                64_p,
                64_p,
            },
        },
        DataType::FLOAT,
    };

    tensor_guid_t input_tensor = b.create_input(input_shape, CreateGrad::NO);

    SUBCASE("num_bottleneck_blocks = 3") {
      SUBCASE("shortcut applicable") {
        tensor_guid_t result = create_yolov10_c2f_module(
            /*cgb=*/b,
            /*input_tensor=*/input_tensor,
            /*num_input_channels=*/128_p,
            /*num_output_channels=*/128_p,
            /*num_bottleneck_blocks=*/3_p,
            /*use_shortcut_connection=*/true,
            /*groups=*/1_p,
            /*expansion_ratio=*/0.5f);

        SUBCASE("contains expected number of operators") {
          std::map<OperatorType, positive_int> result_op_type_counts =
              operator_type_counts_in_computation_graph(b.computation_graph);

          std::map<OperatorType, positive_int> correct_op_type_counts = {
              {OperatorType::INPUT, 1_p},
              {OperatorType::CONV2D, 8_p},
              {OperatorType::WEIGHT, 24_p},
              {OperatorType::BATCHNORM, 8_p},
              {OperatorType::SILU, 8_p},
              {OperatorType::SPLIT, 1_p},
              {OperatorType::CONCAT, 1_p},
              {OperatorType::EW_ADD, 3_p},
          };

          CHECK(result_op_type_counts == correct_op_type_counts);
        }
      }

      SUBCASE("shortcut not applicable") {
        tensor_guid_t result = create_yolov10_c2f_module(
            /*cgb=*/b,
            /*input_tensor=*/input_tensor,
            /*num_input_channels=*/128_p,
            /*num_output_channels=*/128_p,
            /*num_bottleneck_blocks=*/3_p,
            /*use_shortcut_connection=*/false,
            /*groups=*/1_p,
            /*expansion_ratio=*/0.5f);

        SUBCASE("contains expected number of operators") {
          std::map<OperatorType, positive_int> result_op_type_counts =
              operator_type_counts_in_computation_graph(b.computation_graph);

          std::map<OperatorType, positive_int> correct_op_type_counts = {
              {OperatorType::INPUT, 1_p},
              {OperatorType::CONV2D, 8_p},
              {OperatorType::WEIGHT, 24_p},
              {OperatorType::BATCHNORM, 8_p},
              {OperatorType::SILU, 8_p},
              {OperatorType::SPLIT, 1_p},
              {OperatorType::CONCAT, 1_p},
          };

          CHECK(result_op_type_counts == correct_op_type_counts);
        }
      }
    }

    SUBCASE("num_bottleneck_blocks = 6") {
      SUBCASE("shortcut applicable") {
        tensor_guid_t result = create_yolov10_c2f_module(
            /*cgb=*/b,
            /*input_tensor=*/input_tensor,
            /*num_input_channels=*/128_p,
            /*num_output_channels=*/128_p,
            /*num_bottleneck_blocks=*/6_p,
            /*use_shortcut_connection=*/true,
            /*groups=*/1_p,
            /*expansion_ratio=*/0.5f);

        SUBCASE("contains expected number of operators") {
          std::map<OperatorType, positive_int> result_op_type_counts =
              operator_type_counts_in_computation_graph(b.computation_graph);

          std::map<OperatorType, positive_int> correct_op_type_counts = {
              {OperatorType::INPUT, 1_p},
              {OperatorType::CONV2D, 14_p},
              {OperatorType::WEIGHT, 42_p},
              {OperatorType::BATCHNORM, 14_p},
              {OperatorType::SILU, 14_p},
              {OperatorType::SPLIT, 1_p},
              {OperatorType::CONCAT, 1_p},
              {OperatorType::EW_ADD, 6_p},
          };

          CHECK(result_op_type_counts == correct_op_type_counts);
        }
      }
    }

    SUBCASE("num_bottleneck_blocks = 5") {
      SUBCASE("shortcut not applicable") {
        tensor_guid_t result = create_yolov10_c2f_module(
            /*cgb=*/b,
            /*input_tensor=*/input_tensor,
            /*num_input_channels=*/128_p,
            /*num_output_channels=*/48_p,
            /*num_bottleneck_blocks=*/5_p,
            /*use_shortcut_connection=*/false,
            /*groups=*/1_p,
            /*expansion_ratio=*/0.5f);

        SUBCASE("produces correct output shape") {
          TensorShape result_output_shape = b.get_shape(result);

          TensorShape correct_output_shape = TensorShape{
              TensorDims{
                  FFOrdered<positive_int>{
                      2_p,
                      48_p,
                      64_p,
                      64_p,
                  },
              },
              DataType::FLOAT,
          };

          CHECK(result_output_shape == correct_output_shape);
        }

        SUBCASE("contains expected number of operators") {
          std::map<OperatorType, positive_int> result_op_type_counts =
              operator_type_counts_in_computation_graph(b.computation_graph);

          std::map<OperatorType, positive_int> correct_op_type_counts = {
              {OperatorType::INPUT, 1_p},
              {OperatorType::CONV2D, 12_p},
              {OperatorType::WEIGHT, 36_p},
              {OperatorType::BATCHNORM, 12_p},
              {OperatorType::SILU, 12_p},
              {OperatorType::SPLIT, 1_p},
              {OperatorType::CONCAT, 1_p},
          };

          CHECK(result_op_type_counts == correct_op_type_counts);
        }
      }

      SUBCASE("shortcut applicable") {
        tensor_guid_t result = create_yolov10_c2f_module(
            /*cgb=*/b,
            /*input_tensor=*/input_tensor,
            /*num_input_channels=*/128_p,
            /*num_output_channels=*/128_p,
            /*num_bottleneck_blocks=*/5_p,
            /*use_shortcut_connection=*/true,
            /*groups=*/1_p,
            /*expansion_ratio=*/0.5f);

        SUBCASE("produces correct output shape") {
          TensorShape result_output_shape = b.get_shape(result);

          TensorShape correct_output_shape = TensorShape{
              TensorDims{
                  FFOrdered<positive_int>{
                      2_p,
                      128_p,
                      64_p,
                      64_p,
                  },
              },
              DataType::FLOAT,
          };

          CHECK(result_output_shape == correct_output_shape);
        }

        SUBCASE("contains expected number of operators") {
          std::map<OperatorType, positive_int> result_op_type_counts =
              operator_type_counts_in_computation_graph(b.computation_graph);

          std::map<OperatorType, positive_int> correct_op_type_counts = {
              {OperatorType::INPUT, 1_p},
              {OperatorType::CONV2D, 12_p},
              {OperatorType::WEIGHT, 36_p},
              {OperatorType::BATCHNORM, 12_p},
              {OperatorType::SILU, 12_p},
              {OperatorType::SPLIT, 1_p},
              {OperatorType::CONCAT, 1_p},
              {OperatorType::EW_ADD, 5_p},
          };

          CHECK(result_op_type_counts == correct_op_type_counts);
        }
      }
    }
  }

  TEST_CASE("create_yolov10_cib_module") {
    ComputationGraphBuilder b;
    TensorShape input_shape = TensorShape{
        TensorDims{
            FFOrdered<positive_int>{
                2_p,
                128_p,
                64_p,
                64_p,
            },
        },
        DataType::FLOAT,
    };

    tensor_guid_t input_tensor = b.create_input(input_shape, CreateGrad::NO);

    tensor_guid_t result = create_yolov10_cib_module(
        /*cgb=*/b,
        /*input_tensor=*/input_tensor,
        /*num_input_channels=*/128_p,
        /*num_output_channels=*/48_p,
        /*use_shortcut_connection=*/true,
        /*use_large_kernel=*/false,
        /*expansion_ratio=*/0.5f);

    SUBCASE("produces correct output shape") {
      TensorShape result_output_shape = b.get_shape(result);

      TensorShape correct_output_shape = TensorShape{
          TensorDims{
              FFOrdered<positive_int>{
                  2_p,
                  48_p,
                  64_p,
                  64_p,
              },
          },
          DataType::FLOAT,
      };

      CHECK(result_output_shape == correct_output_shape);
    }

    SUBCASE("contains expected number of operators") {
      std::map<OperatorType, positive_int> result_op_type_counts =
          operator_type_counts_in_computation_graph(b.computation_graph);

      std::map<OperatorType, positive_int> correct_op_type_counts = {
          {OperatorType::INPUT, 1_p},
          {OperatorType::CONV2D, 5_p},
          {OperatorType::BATCHNORM, 5_p},
          {OperatorType::SILU, 5_p},
          {OperatorType::WEIGHT, 15_p},
      };

      CHECK(result_op_type_counts == correct_op_type_counts);
    }
  }

  TEST_CASE("create_yolov10_c2fcib_module") {
    ComputationGraphBuilder b;
    TensorShape input_shape = TensorShape{
        TensorDims{
            FFOrdered<positive_int>{
                2_p,
                128_p,
                64_p,
                64_p,
            },
        },
        DataType::FLOAT,
    };

    tensor_guid_t input_tensor = b.create_input(input_shape, CreateGrad::NO);

    SUBCASE("num_cib_modules_to_stack = 3, shortcut applicable") {
      tensor_guid_t result = create_yolov10_c2fcib_module(
          /*cgb=*/b,
          /*input_tensor=*/input_tensor,
          /*num_input_channels=*/128_p,
          /*num_output_channels=*/48_p,
          /*num_cib_modules_to_stack=*/3_p,
          /*use_shortcut_connection=*/true,
          /*use_large_kernel=*/false,
          /*groups=*/1_p,
          /*expansion_ratio=*/0.5f);

      SUBCASE("contains expected number of operators") {
        std::map<OperatorType, positive_int> result_op_type_counts =
            operator_type_counts_in_computation_graph(b.computation_graph);

        std::map<OperatorType, positive_int> correct_op_type_counts = {
            {OperatorType::INPUT, 1_p},
            {OperatorType::CONV2D, 17_p},
            {OperatorType::BATCHNORM, 17_p},
            {OperatorType::SILU, 17_p},
            {OperatorType::WEIGHT, 51_p},
            {OperatorType::SPLIT, 1_p},
            {OperatorType::CONCAT, 1_p},
            {OperatorType::EW_ADD, 3_p},
        };

        CHECK(result_op_type_counts == correct_op_type_counts);
      }
    }

    SUBCASE("num_cib_modules_to_stack = 6, shortcut applicable") {
      tensor_guid_t result = create_yolov10_c2fcib_module(
          /*cgb=*/b,
          /*input_tensor=*/input_tensor,
          /*num_input_channels=*/128_p,
          /*num_output_channels=*/48_p,
          /*num_cib_modules_to_stack=*/6_p,
          /*use_shortcut_connection=*/true,
          /*use_large_kernel=*/false,
          /*groups=*/1_p,
          /*expansion_ratio=*/0.5f);

      SUBCASE("contains expected number of operators") {
        std::map<OperatorType, positive_int> result_op_type_counts =
            operator_type_counts_in_computation_graph(b.computation_graph);

        std::map<OperatorType, positive_int> correct_op_type_counts = {
            {OperatorType::INPUT, 1_p},
            {OperatorType::CONV2D, 32_p},
            {OperatorType::BATCHNORM, 32_p},
            {OperatorType::SILU, 32_p},
            {OperatorType::WEIGHT, 96_p},
            {OperatorType::SPLIT, 1_p},
            {OperatorType::CONCAT, 1_p},
            {OperatorType::EW_ADD, 6_p},
        };

        CHECK(result_op_type_counts == correct_op_type_counts);
      }
    }

    SUBCASE("num_cib_modules_to_stack = 7") {
      SUBCASE("shortcut is enabled") {
        tensor_guid_t result = create_yolov10_c2fcib_module(
            /*cgb=*/b,
            /*input_tensor=*/input_tensor,
            /*num_input_channels=*/128_p,
            /*num_output_channels=*/48_p,
            /*num_cib_modules_to_stack=*/7_p,
            /*use_shortcut_connection=*/true,
            /*use_large_kernel=*/false,
            /*groups=*/1_p,
            /*expansion_ratio=*/0.5f);

        SUBCASE("produces correct output shape") {
          TensorShape result_output_shape = b.get_shape(result);

          TensorShape correct_output_shape = TensorShape{
              TensorDims{
                  FFOrdered<positive_int>{
                      2_p,
                      48_p,
                      64_p,
                      64_p,
                  },
              },
              DataType::FLOAT,
          };

          CHECK(result_output_shape == correct_output_shape);
        }

        SUBCASE("contains expected number of operators") {
          std::map<OperatorType, positive_int> result_op_type_counts =
              operator_type_counts_in_computation_graph(b.computation_graph);

          std::map<OperatorType, positive_int> correct_op_type_counts = {
              {OperatorType::INPUT, 1_p},
              {OperatorType::CONV2D, 37_p},
              {OperatorType::BATCHNORM, 37_p},
              {OperatorType::SILU, 37_p},
              {OperatorType::WEIGHT, 111_p},
              {OperatorType::SPLIT, 1_p},
              {OperatorType::CONCAT, 1_p},
              {OperatorType::EW_ADD, 7_p},
          };

          CHECK(result_op_type_counts == correct_op_type_counts);
        }
      }

      SUBCASE("shortcut is disabled") {
        tensor_guid_t result = create_yolov10_c2fcib_module(
            /*cgb=*/b,
            /*input_tensor=*/input_tensor,
            /*num_input_channels=*/128_p,
            /*num_output_channels=*/48_p,
            /*num_cib_modules_to_stack=*/7_p,
            /*use_shortcut_connection=*/false,
            /*use_large_kernel=*/false,
            /*groups=*/1_p,
            /*expansion_ratio=*/0.5f);

        SUBCASE("produces correct output shape") {
          TensorShape result_output_shape = b.get_shape(result);

          TensorShape correct_output_shape = TensorShape{
              TensorDims{
                  FFOrdered<positive_int>{
                      2_p,
                      48_p,
                      64_p,
                      64_p,
                  },
              },
              DataType::FLOAT,
          };

          CHECK(result_output_shape == correct_output_shape);
        }

        SUBCASE("contains expected number of operators") {
          std::map<OperatorType, positive_int> result_op_type_counts =
              operator_type_counts_in_computation_graph(b.computation_graph);

          std::map<OperatorType, positive_int> correct_op_type_counts = {
              {OperatorType::INPUT, 1_p},
              {OperatorType::CONV2D, 37_p},
              {OperatorType::BATCHNORM, 37_p},
              {OperatorType::SILU, 37_p},
              {OperatorType::WEIGHT, 111_p},
              {OperatorType::SPLIT, 1_p},
              {OperatorType::CONCAT, 1_p},
          };

          CHECK(result_op_type_counts == correct_op_type_counts);
        }
      }
    }
  }

  TEST_CASE("create_yolov10_attention_module") {
    ComputationGraphBuilder b;
    TensorShape input_shape = TensorShape{
        TensorDims{
            FFOrdered<positive_int>{
                2_p,
                128_p,
                64_p,
                64_p,
            },
        },
        DataType::FLOAT,
    };

    tensor_guid_t input_tensor = b.create_input(input_shape, CreateGrad::NO);

    tensor_guid_t result = create_yolov10_attention_module(
        /*cgb=*/b,
        /*input_tensor=*/input_tensor,
        /*num_input_channels=*/128_p,
        /*num_heads=*/8_p,
        /*attn_ratio=*/0.5f);

    SUBCASE("produces correct output shape") {
      TensorShape result_output_shape = b.get_shape(result);

      TensorShape correct_output_shape = input_shape;

      CHECK(result_output_shape == correct_output_shape);
    }

    SUBCASE("contains expected numbers of operators") {
      std::map<OperatorType, positive_int> result_op_type_counts =
          operator_type_counts_in_computation_graph(b.computation_graph);

      std::map<OperatorType, positive_int> correct_op_type_counts = {
          {OperatorType::INPUT, 1_p},
          {OperatorType::CONV2D, 3_p},
          {OperatorType::BATCHNORM, 3_p},
          {OperatorType::WEIGHT, 9_p},
          {OperatorType::RESHAPE, 3_p},
          {OperatorType::SPLIT, 1_p},
          {OperatorType::SCALAR_MULTIPLY, 1_p},
          {OperatorType::TRANSPOSE, 2_p},
          {OperatorType::BATCHMATMUL, 2_p},
          {OperatorType::SOFTMAX, 1_p},
          {OperatorType::EW_ADD, 1_p},
      };

      CHECK(result_op_type_counts == correct_op_type_counts);
    }
  }

  TEST_CASE("create_yolov10_psa_module") {
    // example tensor shapes pulled from
    // https://github.com/ultralytics/ultralytics/blob/f8ad132a15b5f6818c2ce0647b40dc57e993bf0c/ultralytics/nn/modules/block.py#L1398-L1401

    ComputationGraphBuilder b;
    TensorShape input_shape = TensorShape{
        TensorDims{
            FFOrdered<positive_int>{
                1_p,
                128_p,
                64_p,
                64_p,
            },
        },
        DataType::FLOAT,
    };

    tensor_guid_t input_tensor = b.create_input(input_shape, CreateGrad::NO);

    tensor_guid_t result = create_yolov10_psa_module(
        /*cgb=*/b,
        /*input_tensor=*/input_tensor,
        /*num_input_channels=*/128_p,
        /*num_output_channels=*/128_p,
        /*expansion_ratio=*/0.5f);

    SUBCASE("produces correct output shape") {
      TensorShape result_output_shape = b.get_shape(result);

      TensorShape correct_output_shape = input_shape;

      CHECK(result_output_shape == correct_output_shape);
    }

    SUBCASE("contains expected numbers of operators") {
      std::map<OperatorType, positive_int> result_op_type_counts =
          operator_type_counts_in_computation_graph(b.computation_graph);

      std::map<OperatorType, positive_int> correct_op_type_counts = {
          {OperatorType::INPUT, 1_p},
          {OperatorType::CONV2D, 7_p},
          {OperatorType::BATCHNORM, 7_p},
          {OperatorType::WEIGHT, 21_p},
          {OperatorType::RESHAPE, 3_p},
          {OperatorType::SPLIT, 2_p},
          {OperatorType::SCALAR_MULTIPLY, 1_p},
          {OperatorType::TRANSPOSE, 2_p},
          {OperatorType::BATCHMATMUL, 2_p},
          {OperatorType::SOFTMAX, 1_p},
          {OperatorType::EW_ADD, 3_p},
          {OperatorType::CONCAT, 1_p},
          {OperatorType::SILU, 3_p},
      };

      CHECK(result_op_type_counts == correct_op_type_counts);
    }
  }

  TEST_CASE("create_yolov10_bottleneck_module") {
    ComputationGraphBuilder b;
    TensorShape input_shape = TensorShape{
        TensorDims{
            FFOrdered<positive_int>{
                1_p,
                128_p,
                64_p,
                64_p,
            },
        },
        DataType::FLOAT,
    };

    tensor_guid_t input_tensor = b.create_input(input_shape, CreateGrad::NO);

    SUBCASE("shortcut is not applicable") {
      tensor_guid_t result = create_yolov10_bottleneck_module(
          /*cgb=*/b,
          /*input_tensor=*/input_tensor,
          /*num_input_channels=*/128_p,
          /*num_output_channels=*/96_p,
          /*use_shortcut_connection=*/false,
          /*groups=*/1_p,
          /*kernel_size_1=*/5_p,
          /*kernel_size_2=*/7_p,
          /*expansion_ratio=*/0.5f);

      SUBCASE("produces correct output shape") {
        TensorShape result_output_shape = b.get_shape(result);

        TensorShape correct_output_shape = TensorShape{
            TensorDims{
                FFOrdered<positive_int>{
                    1_p,
                    96_p,
                    64_p,
                    64_p,
                },
            },
            DataType::FLOAT,
        };

        CHECK(result_output_shape == correct_output_shape);
      }

      SUBCASE("contains expected numbers of operators") {
        std::map<OperatorType, positive_int> result_op_type_counts =
            operator_type_counts_in_computation_graph(b.computation_graph);

        std::map<OperatorType, positive_int> correct_op_type_counts = {
            {OperatorType::INPUT, 1_p},
            {OperatorType::CONV2D, 2_p},
            {OperatorType::BATCHNORM, 2_p},
            {OperatorType::WEIGHT, 6_p},
            {OperatorType::SILU, 2_p},
        };

        CHECK(result_op_type_counts == correct_op_type_counts);
      }
    }

    SUBCASE("shortcut is applicable") {
      tensor_guid_t result = create_yolov10_bottleneck_module(
          /*cgb=*/b,
          /*input_tensor=*/input_tensor,
          /*num_input_channels=*/128_p,
          /*num_output_channels=*/128_p,
          /*use_shortcut_connection=*/true,
          /*groups=*/1_p,
          /*kernel_size_1=*/5_p,
          /*kernel_size_2=*/7_p,
          /*expansion_ratio=*/0.5f);

      SUBCASE("produces correct output shape") {
        TensorShape result_output_shape = b.get_shape(result);

        TensorShape correct_output_shape = input_shape;

        CHECK(result_output_shape == correct_output_shape);
      }

      SUBCASE("contains expected numbers of operators") {
        std::map<OperatorType, positive_int> result_op_type_counts =
            operator_type_counts_in_computation_graph(b.computation_graph);

        std::map<OperatorType, positive_int> correct_op_type_counts = {
            {OperatorType::INPUT, 1_p},
            {OperatorType::CONV2D, 2_p},
            {OperatorType::BATCHNORM, 2_p},
            {OperatorType::WEIGHT, 6_p},
            {OperatorType::SILU, 2_p},
            {OperatorType::EW_ADD, 1_p},
        };

        CHECK(result_op_type_counts == correct_op_type_counts);
      }
    }
  }

  TEST_CASE("create_yolov10_sppf_module") {
    ComputationGraphBuilder b;
    TensorShape input_shape = TensorShape{
        TensorDims{
            FFOrdered<positive_int>{
                1_p,
                128_p,
                64_p,
                64_p,
            },
        },
        DataType::FLOAT,
    };

    tensor_guid_t input_tensor = b.create_input(input_shape, CreateGrad::NO);

    SUBCASE("shortcut not applicable") {
      tensor_guid_t result = create_yolov10_sppf_module(
          /*cgb=*/b,
          /*input_tensor=*/input_tensor,
          /*num_input_channels=*/128_p,
          /*num_output_channels=*/96_p,
          /*kernel_size=*/7_p,
          /*num_pooling_iterations=*/4_p,
          /*use_shortcut_connection=*/true);

      SUBCASE("produces correct output shape") {
        TensorShape result_output_shape = b.get_shape(result);

        TensorShape correct_output_shape = TensorShape{
            TensorDims{
                FFOrdered<positive_int>{
                    1_p,
                    96_p,
                    64_p,
                    64_p,
                },
            },
            DataType::FLOAT,
        };

        CHECK(result_output_shape == correct_output_shape);
      }

      SUBCASE("contains expected numbers of operators") {
        std::map<OperatorType, positive_int> result_op_type_counts =
            operator_type_counts_in_computation_graph(b.computation_graph);

        std::map<OperatorType, positive_int> correct_op_type_counts = {
            {OperatorType::INPUT, 1_p},
            {OperatorType::WEIGHT, 6_p},
            {OperatorType::BATCHNORM, 2_p},
            {OperatorType::CONV2D, 2_p},
            {OperatorType::POOL2D, 4_p},
            {OperatorType::CONCAT, 1_p},
            {OperatorType::SILU, 1_p},
        };

        CHECK(result_op_type_counts == correct_op_type_counts);
      }
    }

    SUBCASE("shortcut applicable") {
      SUBCASE("num_pooling_iterations = 3") {
        tensor_guid_t result = create_yolov10_sppf_module(
            /*cgb=*/b,
            /*input_tensor=*/input_tensor,
            /*num_input_channels=*/128_p,
            /*num_output_channels=*/128_p,
            /*kernel_size=*/7_p,
            /*num_pooling_iterations=*/3_p,
            /*use_shortcut_connection=*/true);

        SUBCASE("contains expected numbers of operators") {
          std::map<OperatorType, positive_int> result_op_type_counts =
              operator_type_counts_in_computation_graph(b.computation_graph);

          std::map<OperatorType, positive_int> correct_op_type_counts = {
              {OperatorType::INPUT, 1_p},
              {OperatorType::WEIGHT, 6_p},
              {OperatorType::BATCHNORM, 2_p},
              {OperatorType::CONV2D, 2_p},
              {OperatorType::POOL2D, 3_p},
              {OperatorType::CONCAT, 1_p},
              {OperatorType::SILU, 1_p},
              {OperatorType::EW_ADD, 1_p},
          };

          CHECK(result_op_type_counts == correct_op_type_counts);
        }
      }

      SUBCASE("num_pooling_iterations = 4") {
        tensor_guid_t result = create_yolov10_sppf_module(
            /*cgb=*/b,
            /*input_tensor=*/input_tensor,
            /*num_input_channels=*/128_p,
            /*num_output_channels=*/128_p,
            /*kernel_size=*/7_p,
            /*num_pooling_iterations=*/4_p,
            /*use_shortcut_connection=*/true);

        SUBCASE("produces correct output shape") {
          TensorShape result_output_shape = b.get_shape(result);
          TensorShape correct_output_shape = input_shape;

          CHECK(result_output_shape == correct_output_shape);
        }

        SUBCASE("contains expected numbers of operators") {
          std::map<OperatorType, positive_int> result_op_type_counts =
              operator_type_counts_in_computation_graph(b.computation_graph);

          std::map<OperatorType, positive_int> correct_op_type_counts = {
              {OperatorType::INPUT, 1_p},
              {OperatorType::WEIGHT, 6_p},
              {OperatorType::BATCHNORM, 2_p},
              {OperatorType::CONV2D, 2_p},
              {OperatorType::POOL2D, 4_p},
              {OperatorType::CONCAT, 1_p},
              {OperatorType::SILU, 1_p},
              {OperatorType::EW_ADD, 1_p},
          };

          CHECK(result_op_type_counts == correct_op_type_counts);
        }
      }
    }
  }

  TEST_CASE("create_yolov10_layer") {
    ComputationGraphBuilder b;

    TensorShape input_shape = TensorShape{
        TensorDims{
            FFOrdered<positive_int>{
                1_p,
                128_p,
                64_p,
                64_p,
            },
        },
        DataType::FLOAT,
    };

    tensor_guid_t input_tensor = b.create_input(input_shape, CreateGrad::NO);

    SUBCASE("depth scaling is applied") {
      SUBCASE("if scaling factor is 1, num repeats stays the same") {
        create_yolov10_layer(
            /*cgb=*/b,
            /*layer_config=*/
            YOLOv10LayerConfig{
                YOLOv10LayerConfigC2f{
                    /*input_tensor_idx=*/yolov10_tensor_idx_t{-1},
                    /*num_output_channels=*/256_p,
                    /*num_bottleneck_blocks=*/3_p,
                    /*use_shortcut_connection=*/false,
                },
            },
            /*num_classes=*/64_p,
            /*scaling_config=*/
            YOLOv10ScalingConfig{
                /*depth_scaling_factor=*/1.0f,
                /*width_scaling_factor=*/1.0f,
                /*max_channels=*/512_p,
            },
            /*past_layer_outputs=*/
            {
                input_tensor,
            });

        std::map<OperatorType, positive_int> result_op_type_counts =
            operator_type_counts_in_computation_graph(b.computation_graph);

        CHECK(result_op_type_counts.at(OperatorType::CONV2D) == 8_p);
      }

      SUBCASE("if scaling factor is big, num repeats increases") {
        create_yolov10_layer(
            /*cgb=*/b,
            /*layer_config=*/
            YOLOv10LayerConfig{
                YOLOv10LayerConfigC2f{
                    /*input_tensor_idx=*/yolov10_tensor_idx_t{-1},
                    /*num_output_channels=*/256_p,
                    /*num_bottleneck_blocks=*/3_p,
                    /*use_shortcut_connection=*/false,
                },
            },
            /*num_classes=*/64_p,
            /*scaling_config=*/
            YOLOv10ScalingConfig{
                /*depth_scaling_factor=*/4.0f,
                /*width_scaling_factor=*/1.0f,
                /*max_channels=*/512_p,
            },
            /*past_layer_outputs=*/
            {
                input_tensor,
            });

        std::map<OperatorType, positive_int> result_op_type_counts =
            operator_type_counts_in_computation_graph(b.computation_graph);

        CHECK(result_op_type_counts.at(OperatorType::CONV2D) == 26_p);
      }
    }

    SUBCASE("width scaling is applied") {
      SUBCASE("if scaling factor is 1, num_output_channels stays the same") {
        create_yolov10_layer(
            /*cgb=*/b,
            /*layer_config=*/
            YOLOv10LayerConfig{
                YOLOv10LayerConfigConv{
                    /*input_tensor_idx=*/yolov10_tensor_idx_t{-1},
                    /*num_output_channels=*/256_p,
                    /*kernel_size=*/3_p,
                    /*stride=*/1_p,
                },
            },
            /*num_classes=*/64_p,
            /*scaling_config=*/
            YOLOv10ScalingConfig{
                /*depth_scaling_factor=*/1.0f,
                /*width_scaling_factor=*/1.0f,
                /*max_channels=*/512_p,
            },
            /*past_layer_outputs=*/
            {
                input_tensor,
            });

        Conv2DAttrs conv_op_attrs = get_only(
            filtrans(values(get_layer_attrs_mapping(b.computation_graph)),
                     [&](LayerAttrs const &a) -> std::optional<Conv2DAttrs> {
                       return a.op_attrs.try_require_conv2d();
                     }));

        CHECK(conv_op_attrs.out_channels == 256_p);
      }

      SUBCASE("if scaling factor is big, num_output_channels increases") {
        create_yolov10_layer(
            /*cgb=*/b,
            /*layer_config=*/
            YOLOv10LayerConfig{
                YOLOv10LayerConfigConv{
                    /*input_tensor_idx=*/yolov10_tensor_idx_t{-1},
                    /*num_output_channels=*/256_p,
                    /*kernel_size=*/3_p,
                    /*stride=*/1_p,
                },
            },
            /*num_classes=*/64_p,
            /*scaling_config=*/
            YOLOv10ScalingConfig{
                /*depth_scaling_factor=*/1.0f,
                /*width_scaling_factor=*/5.0f,
                /*max_channels=*/512_p,
            },
            /*past_layer_outputs=*/
            {
                input_tensor,
            });

        Conv2DAttrs conv_op_attrs = get_only(
            filtrans(values(get_layer_attrs_mapping(b.computation_graph)),
                     [&](LayerAttrs const &a) -> std::optional<Conv2DAttrs> {
                       return a.op_attrs.try_require_conv2d();
                     }));

        CHECK(conv_op_attrs.out_channels == 1280_p);
      }
    }

    SUBCASE("max_channels is applied") {
      SUBCASE("if num_output_channels <= max_channels, num_output_channels is "
              "used") {
        create_yolov10_layer(
            /*cgb=*/b,
            /*layer_config=*/
            YOLOv10LayerConfig{
                YOLOv10LayerConfigConv{
                    /*input_tensor_idx=*/yolov10_tensor_idx_t{-1},
                    /*num_output_channels=*/256_p,
                    /*kernel_size=*/3_p,
                    /*stride=*/1_p,
                },
            },
            /*num_classes=*/64_p,
            /*scaling_config=*/
            YOLOv10ScalingConfig{
                /*depth_scaling_factor=*/1.0f,
                /*width_scaling_factor=*/1.0f,
                /*max_channels=*/512_p,
            },
            /*past_layer_outputs=*/
            {
                input_tensor,
            });

        Conv2DAttrs conv_op_attrs = get_only(
            filtrans(values(get_layer_attrs_mapping(b.computation_graph)),
                     [&](LayerAttrs const &a) -> std::optional<Conv2DAttrs> {
                       return a.op_attrs.try_require_conv2d();
                     }));

        CHECK(conv_op_attrs.out_channels == 256_p);
      }

      SUBCASE("if num_output_channels > max_channels, max_channels is used") {
        create_yolov10_layer(
            /*cgb=*/b,
            /*layer_config=*/
            YOLOv10LayerConfig{
                YOLOv10LayerConfigConv{
                    /*input_tensor_idx=*/yolov10_tensor_idx_t{-1},
                    /*num_output_channels=*/256_p,
                    /*kernel_size=*/3_p,
                    /*stride=*/1_p,
                },
            },
            /*num_classes=*/64_p,
            /*scaling_config=*/
            YOLOv10ScalingConfig{
                /*depth_scaling_factor=*/1.0f,
                /*width_scaling_factor=*/1.0f,
                /*max_channels=*/32_p,
            },
            /*past_layer_outputs=*/
            {
                input_tensor,
            });

        Conv2DAttrs conv_op_attrs = get_only(
            filtrans(values(get_layer_attrs_mapping(b.computation_graph)),
                     [&](LayerAttrs const &a) -> std::optional<Conv2DAttrs> {
                       return a.op_attrs.try_require_conv2d();
                     }));

        CHECK(conv_op_attrs.out_channels == 32_p);
      }
    }
  }

  TEST_CASE("get_yolov10_computation_graph") {
    std::set<TensorShape> correct_output_shapes = {
        TensorShape{
            TensorDims{
                FFOrdered<positive_int>{
                    64_p,
                    64_p,
                    8400_p,
                },
            },
            DataType::FLOAT,
        },
        TensorShape{
            TensorDims{
                FFOrdered<positive_int>{
                    64_p,
                    80_p,
                    8400_p,
                },
            },
            DataType::FLOAT,
        },
    };

    auto get_tensor_shapes =
        [](ComputationGraph const &cg,
           std::set<tensor_guid_t> const &ts) -> std::set<TensorShape> {
      return transform(ts, [&](tensor_guid_t t) -> TensorShape {
        return get_tensor_attrs(cg, t).shape;
      });
    };

    SUBCASE("yolov10x") {
      YOLOv10Config config = get_yolov10x_config(
          /*batch_size=*/64_p,
          /*end2end=*/false,
          /*image_height=*/640_p,
          /*image_width=*/640_p);

      ComputationGraph result = get_yolov10_computation_graph(config);

      SUBCASE("contains expected number of operators") {
        std::map<OperatorType, positive_int> result_op_type_counts =
            operator_type_counts_in_computation_graph(result);

        auto multiply_count =
            [](positive_int factor,
               std::map<OperatorType, positive_int> const &count) {
              return map_values(count, [&](positive_int x) -> positive_int {
                return x * factor;
              });
            };

        auto binary_add_counts =
            [](std::map<OperatorType, positive_int> const &lhs,
               std::map<OperatorType, positive_int> const &rhs)
            -> std::map<OperatorType, positive_int> {
          std::set<OperatorType> all_keys = set_union(keys(lhs), keys(rhs));

          return generate_map(all_keys, [&](OperatorType o) -> positive_int {
            nonnegative_int result = 0_n;
            if (contains_key(lhs, o)) {
              result += lhs.at(o);
            };

            if (contains_key(rhs, o)) {
              result += rhs.at(o);
            };

            return positive_int{result};
          });
        };

        auto add_counts =
            [&](std::vector<std::map<OperatorType, positive_int>> const &counts)
            -> std::map<OperatorType, positive_int> {
          return foldl(counts,
                       std::map<OperatorType, positive_int>{},
                       binary_add_counts);
        };

        std::map<OperatorType, positive_int> correct_op_type_counts = [&]() {
          std::map<OperatorType, positive_int> single_input_op_count = {
              {OperatorType::INPUT, 1_p},
          };

          std::map<OperatorType, positive_int>
              single_ultralytics_conv_op_count = {
                  {OperatorType::CONV2D, 1_p},
                  {OperatorType::BATCHNORM, 1_p},
                  {OperatorType::SILU, 1_p},
                  {OperatorType::WEIGHT, 3_p},
              };

          std::map<OperatorType, positive_int> single_scdown_op_count = {
              {OperatorType::CONV2D, 2_p},
              {OperatorType::BATCHNORM, 2_p},
              {OperatorType::SILU, 1_p},
              {OperatorType::WEIGHT, 6_p},
          };

          std::map<OperatorType, positive_int> single_sppf_op_count = {
              {OperatorType::WEIGHT, 6_p},
              {OperatorType::BATCHNORM, 2_p},
              {OperatorType::CONV2D, 2_p},
              {OperatorType::POOL2D, 3_p},
              {OperatorType::CONCAT, 1_p},
              {OperatorType::SILU, 1_p},
          };

          std::map<OperatorType, positive_int> single_psa_op_count = {
              {OperatorType::CONV2D, 7_p},
              {OperatorType::BATCHNORM, 7_p},
              {OperatorType::WEIGHT, 21_p},
              {OperatorType::RESHAPE, 3_p},
              {OperatorType::SPLIT, 2_p},
              {OperatorType::SCALAR_MULTIPLY, 1_p},
              {OperatorType::TRANSPOSE, 2_p},
              {OperatorType::BATCHMATMUL, 2_p},
              {OperatorType::SOFTMAX, 1_p},
              {OperatorType::EW_ADD, 3_p},
              {OperatorType::CONCAT, 1_p},
              {OperatorType::SILU, 3_p},
          };

          std::map<OperatorType, positive_int> single_upsample_op_count = {
              {OperatorType::UPSAMPLE, 1_p},
          };

          std::map<OperatorType, positive_int> single_concat_op_count = {
              {OperatorType::CONCAT, 1_p},
          };

          std::map<OperatorType, positive_int> single_c2f_3_t_op_count = {
              {OperatorType::CONV2D, 8_p},
              {OperatorType::WEIGHT, 24_p},
              {OperatorType::BATCHNORM, 8_p},
              {OperatorType::SILU, 8_p},
              {OperatorType::SPLIT, 1_p},
              {OperatorType::CONCAT, 1_p},
              {OperatorType::EW_ADD, 3_p},
          };

          std::map<OperatorType, positive_int> single_c2f_3_f_op_count = {
              {OperatorType::CONV2D, 8_p},
              {OperatorType::WEIGHT, 24_p},
              {OperatorType::BATCHNORM, 8_p},
              {OperatorType::SILU, 8_p},
              {OperatorType::SPLIT, 1_p},
              {OperatorType::CONCAT, 1_p},
          };

          std::map<OperatorType, positive_int> single_c2f_6_t_op_count = {
              {OperatorType::CONV2D, 14_p},
              {OperatorType::WEIGHT, 42_p},
              {OperatorType::BATCHNORM, 14_p},
              {OperatorType::SILU, 14_p},
              {OperatorType::SPLIT, 1_p},
              {OperatorType::CONCAT, 1_p},
              {OperatorType::EW_ADD, 6_p},
          };

          std::map<OperatorType, positive_int> single_c2fcib_3_t_op_count = {
              {OperatorType::CONV2D, 17_p},
              {OperatorType::BATCHNORM, 17_p},
              {OperatorType::SILU, 17_p},
              {OperatorType::WEIGHT, 51_p},
              {OperatorType::SPLIT, 1_p},
              {OperatorType::CONCAT, 1_p},
              {OperatorType::EW_ADD, 3_p},
          };

          std::map<OperatorType, positive_int> single_c2fcib_6_t_op_count = {
              {OperatorType::CONV2D, 32_p},
              {OperatorType::BATCHNORM, 32_p},
              {OperatorType::SILU, 32_p},
              {OperatorType::WEIGHT, 96_p},
              {OperatorType::SPLIT, 1_p},
              {OperatorType::CONCAT, 1_p},
              {OperatorType::EW_ADD, 6_p},
          };

          std::map<OperatorType, positive_int> single_v10detect_op_count = {
              {OperatorType::CONV2D, 24_p},
              {OperatorType::BATCHNORM, 18_p},
              {OperatorType::WEIGHT, 66_p},
              {OperatorType::SILU, 18_p},
              {OperatorType::RESHAPE, 6_p},
              {OperatorType::CONCAT, 2_p},
          };

          return add_counts(std::vector<std::map<OperatorType, positive_int>>{
              single_input_op_count,
              multiply_count(4_p, single_ultralytics_conv_op_count),
              multiply_count(3_p, single_scdown_op_count),
              single_sppf_op_count,
              single_psa_op_count,
              multiply_count(2_p, single_upsample_op_count),
              multiply_count(4_p, single_concat_op_count),
              single_c2f_3_t_op_count,
              single_c2f_3_f_op_count,
              single_c2f_6_t_op_count,
              multiply_count(4_p, single_c2fcib_3_t_op_count),
              single_c2fcib_6_t_op_count,
              single_v10detect_op_count,
          });
        }();

        CHECK(result_op_type_counts == correct_op_type_counts);
      }

      SUBCASE("all values except the two outputs are used") {
        std::set<tensor_guid_t> unused_tensors = cg_get_unused_tensors(result);

        ASSERT(unused_tensors.size() == 2);

        SUBCASE("outputs have correct shape") {
          std::set<TensorShape> output_shapes =
              get_tensor_shapes(result, unused_tensors);

          CHECK(output_shapes == correct_output_shapes);
        }
      }
    }

    SUBCASE("yolov10l") {
      YOLOv10Config config = get_yolov10l_config(
          /*batch_size=*/64_p,
          /*end2end=*/false,
          /*image_height=*/640_p,
          /*image_width=*/640_p);

      ComputationGraph result = get_yolov10_computation_graph(config);

      SUBCASE("all values except the two outputs are used") {
        std::set<tensor_guid_t> unused_tensors = cg_get_unused_tensors(result);

        CHECK(unused_tensors.size() == 2);

        SUBCASE("outputs have correct shape") {
          std::set<TensorShape> output_shapes =
              get_tensor_shapes(result, unused_tensors);

          CHECK(output_shapes == correct_output_shapes);
        }
      }
    }

    SUBCASE("yolov10b") {
      YOLOv10Config config = get_yolov10b_config(
          /*batch_size=*/64_p,
          /*end2end=*/false,
          /*image_height=*/640_p,
          /*image_width=*/640_p);

      ComputationGraph result = get_yolov10_computation_graph(config);

      SUBCASE("all values except the two outputs are used") {
        std::set<tensor_guid_t> unused_tensors = cg_get_unused_tensors(result);

        CHECK(unused_tensors.size() == 2);

        SUBCASE("outputs have correct shape") {
          std::set<TensorShape> output_shapes =
              get_tensor_shapes(result, unused_tensors);

          CHECK(output_shapes == correct_output_shapes);
        }
      }
    }

    SUBCASE("yolov10m") {
      YOLOv10Config config = get_yolov10m_config(
          /*batch_size=*/64_p,
          /*end2end=*/false,
          /*image_height=*/640_p,
          /*image_width=*/640_p);

      ComputationGraph result = get_yolov10_computation_graph(config);

      SUBCASE("all values except the two outputs are used") {
        std::set<tensor_guid_t> unused_tensors = cg_get_unused_tensors(result);

        CHECK(unused_tensors.size() == 2);

        SUBCASE("outputs have correct shape") {
          std::set<TensorShape> output_shapes =
              get_tensor_shapes(result, unused_tensors);

          CHECK(output_shapes == correct_output_shapes);
        }
      }
    }

    SUBCASE("yolov10s") {
      YOLOv10Config config = get_yolov10s_config(
          /*batch_size=*/64_p,
          /*end2end=*/false,
          /*image_height=*/640_p,
          /*image_width=*/640_p);

      ComputationGraph result = get_yolov10_computation_graph(config);

      SUBCASE("all values except the two outputs are used") {
        std::set<tensor_guid_t> unused_tensors = cg_get_unused_tensors(result);

        CHECK(unused_tensors.size() == 2);

        SUBCASE("outputs have correct shape") {
          std::set<TensorShape> output_shapes =
              get_tensor_shapes(result, unused_tensors);

          CHECK(output_shapes == correct_output_shapes);
        }
      }
    }

    SUBCASE("yolov10n") {
      YOLOv10Config config = get_yolov10n_config(
          /*batch_size=*/64_p,
          /*end2end=*/false,
          /*image_height=*/640_p,
          /*image_width=*/640_p);

      ComputationGraph result = get_yolov10_computation_graph(config);

      SUBCASE("all values except the two outputs are used") {
        std::set<tensor_guid_t> unused_tensors = cg_get_unused_tensors(result);

        CHECK(unused_tensors.size() == 2);

        SUBCASE("outputs have correct shape") {
          std::set<TensorShape> output_shapes =
              get_tensor_shapes(result, unused_tensors);

          CHECK(output_shapes == correct_output_shapes);
        }
      }
    }
  }

  TEST_CASE("get_yolov10_config") {
    SUBCASE("all configs are distinct") {
      std::set<YOLOv10Scale> scales = {
          YOLOv10Scale::NANO,
          YOLOv10Scale::SMALL,
          YOLOv10Scale::MEDIUM,
          YOLOv10Scale::BALANCED,
          YOLOv10Scale::LARGE,
          YOLOv10Scale::EXTRA_LARGE,
      };

      std::set<YOLOv10Config> result =
          transform(scales, [&](YOLOv10Scale scale) -> YOLOv10Config {
            return get_yolov10_config(
                /*scale=*/scale,
                /*batch_size=*/16_p,
                /*end2end=*/false);
          });

      CHECK(are_all_distinct(result));
    }
  }
}
