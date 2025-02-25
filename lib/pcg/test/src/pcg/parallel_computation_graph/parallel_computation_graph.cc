#include "pcg/parallel_computation_graph/parallel_computation_graph.h"
#include "op-attrs/ops/element_unary.h"
#include "op-attrs/ops/linear.h"
#include "op-attrs/ops/replicate.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "pcg/parallel_computation_graph/parallel_computation_graph_builder.h"
#include "utils/containers/get_only.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

template <typename T>
static ParallelLayerAttrs make_layer_attrs(T const &op_attrs) {
  return ParallelLayerAttrs{
      /*op_attrs=*/PCGOperatorAttrs{op_attrs},
      /*name=*/std::nullopt,
  };
};

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("topological_ordering") {
    // TODO(@lockshaw) should probably be replaced with a rapidcheck test that
    // compares ParallelComputationGraph to DataflowGraph, but since we
    // currently don't have rapidcheck generation for DataflowGraph this will
    // have to do for now

    ParallelComputationGraph pcg = empty_parallel_computation_graph();

    TensorShape input_shape = TensorShape{
        TensorDims{
            FFOrdered<nonnegative_int>{
                12_n,
                16_n,
            },
        },
        DataType::FLOAT,
    };

    ElementUnaryAttrs relu_attrs = make_relu_attrs();

    ParallelLayerAddedResult layer1_added =
        pcg_add_input_layer(pcg, input_shape);
    parallel_layer_guid_t layer1 = layer1_added.parallel_layer;
    parallel_tensor_guid_t tensor1 = get_only(layer1_added.outputs);

    ParallelLayerAddedResult layer2_added =
        add_parallel_layer(pcg, make_layer_attrs(relu_attrs), {tensor1}, {});
    parallel_layer_guid_t layer2 = layer2_added.parallel_layer;
    parallel_tensor_guid_t tensor2 = get_only(layer2_added.outputs);

    ParallelLayerAddedResult layer3_added =
        add_parallel_layer(pcg, make_layer_attrs(relu_attrs), {tensor2}, {});
    parallel_layer_guid_t layer3 = layer3_added.parallel_layer;
    parallel_tensor_guid_t tensor3 = get_only(layer3_added.outputs);

    std::vector<parallel_layer_guid_t> result = topological_ordering(pcg);
    std::vector<parallel_layer_guid_t> correct = {layer1, layer2, layer3};
    CHECK(result == correct);
  }

  TEST_CASE(
      "get_incoming_inputs(ParallelComputationGraph, parallel_layer_guid_t)") {
    ParallelComputationGraph pcg = empty_parallel_computation_graph();

    TensorShape input_shape = TensorShape{
        TensorDims{
            FFOrdered<nonnegative_int>{10_n, 12_n},
        },
        DataType::FLOAT,
    };

    SUBCASE("layer has no inputs") {
      ParallelLayerAddedResult input_added =
          pcg_add_input_layer(pcg, input_shape);

      std::vector<parallel_tensor_guid_t> result =
          get_incoming_inputs(pcg, input_added.parallel_layer);
      std::vector<parallel_tensor_guid_t> correct = {};

      CHECK(result == correct);
    }

    SUBCASE("layer has inputs and weights") {
      std::string my_op_name = "my op";

      LinearAttrs linear_attrs = LinearAttrs{
          /*out_channels=*/14_n,
          /*use_bias=*/true,
          /*data_type=*/DataType::FLOAT,
          /*activation=*/Activation::RELU,
          /*regularizer=*/std::nullopt,
      };

      WeightAttrs projection_weight_attrs = WeightAttrs{
          /*tensor_shape=*/throw_if_unexpected(
              get_projection_shape(linear_attrs, input_shape)),
          /*initializer=*/InitializerAttrs{ZeroInitializerAttrs{}},
      };

      WeightAttrs bias_weight_attrs = WeightAttrs{
          /*tensor_shape=*/throw_if_unexpected(
              get_bias_shape(linear_attrs, input_shape)),
          /*initializer=*/InitializerAttrs{ZeroInitializerAttrs{}},
      };

      ParallelLayerAddedResult input_added =
          pcg_add_input_layer(pcg, input_shape);
      parallel_tensor_guid_t t_input = get_only(input_added.outputs);

      ParallelLayerAddedResult projection_weight_added = add_parallel_layer(
          pcg, make_layer_attrs(projection_weight_attrs), {}, {});
      parallel_tensor_guid_t t_projection =
          get_only(projection_weight_added.outputs);

      ParallelLayerAddedResult bias_weight_added =
          add_parallel_layer(pcg, make_layer_attrs(bias_weight_attrs), {}, {});
      parallel_tensor_guid_t t_bias = get_only(bias_weight_added.outputs);

      ParallelLayerAddedResult linear_added =
          add_parallel_layer(pcg,
                             make_layer_attrs(linear_attrs),
                             {t_input},
                             {t_projection, t_bias});

      std::vector<parallel_tensor_guid_t> result =
          get_incoming_inputs(pcg, linear_added.parallel_layer);
      std::vector<parallel_tensor_guid_t> correct = {t_input};

      CHECK(result == correct);
    }
  }

  TEST_CASE(
      "get_source_layer(ParallelComputationGraph, parallel_tensor_guid_t)") {
    TensorShape input_shape = TensorShape{
        TensorDims{
            FFOrdered<nonnegative_int>{
                10_n,
                12_n,
            },
        },
        DataType::FLOAT,
    };

    ParallelComputationGraph pcg = empty_parallel_computation_graph();

    ElementUnaryAttrs relu_attrs = make_relu_attrs();

    SUBCASE("single layer") {
      ParallelLayerAddedResult layer1_added =
          pcg_add_input_layer(pcg, input_shape);
      parallel_layer_guid_t layer1 = layer1_added.parallel_layer;
      parallel_tensor_guid_t tensor1 = get_only(layer1_added.outputs);

      parallel_layer_guid_t result = get_source_layer(pcg, tensor1);
      parallel_layer_guid_t correct = layer1;
      CHECK(result == correct);
    }

    SUBCASE("two connected layers") {
      ParallelLayerAddedResult layer1_added =
          pcg_add_input_layer(pcg, input_shape);
      parallel_layer_guid_t layer1 = layer1_added.parallel_layer;
      parallel_tensor_guid_t tensor1 = get_only(layer1_added.outputs);

      ParallelLayerAddedResult layer2_added =
          add_parallel_layer(pcg, make_layer_attrs(relu_attrs), {tensor1}, {});
      parallel_layer_guid_t layer2 = layer2_added.parallel_layer;

      parallel_layer_guid_t result = get_source_layer(pcg, tensor1);
      parallel_layer_guid_t correct = layer1;
      CHECK(result == correct);
    }

    SUBCASE("three layers in series") {
      ParallelLayerAddedResult layer1_added =
          pcg_add_input_layer(pcg, input_shape);
      parallel_layer_guid_t layer1 = layer1_added.parallel_layer;
      parallel_tensor_guid_t tensor1 = get_only(layer1_added.outputs);

      ParallelLayerAddedResult layer2_added =
          add_parallel_layer(pcg, make_layer_attrs(relu_attrs), {tensor1}, {});
      parallel_layer_guid_t layer2 = layer2_added.parallel_layer;
      parallel_tensor_guid_t tensor2 = get_only(layer2_added.outputs);

      ParallelLayerAddedResult layer3_added =
          add_parallel_layer(pcg, make_layer_attrs(relu_attrs), {tensor1}, {});
      parallel_layer_guid_t layer3 = layer3_added.parallel_layer;

      SUBCASE("tensor 1") {
        parallel_layer_guid_t result = get_source_layer(pcg, tensor1);
        parallel_layer_guid_t correct = layer1;
        CHECK(result == correct);
      }

      SUBCASE("tensor 2") {
        parallel_layer_guid_t result = get_source_layer(pcg, tensor2);
        parallel_layer_guid_t correct = layer2;
        CHECK(result == correct);
      }
    }
  }

  TEST_CASE(
      "get_incoming_weights(ParallelComputationGraph, parallel_layer_guid_t)") {
    TensorShape input_shape = TensorShape{
        TensorDims{
            FFOrdered<nonnegative_int>{
                10_n,
                12_n,
            },
        },
        DataType::FLOAT,
    };

    ParallelComputationGraph pcg = empty_parallel_computation_graph();

    SUBCASE("layer has no inputs or weights") {
      ParallelLayerAddedResult input_added =
          pcg_add_input_layer(pcg, input_shape);

      std::vector<parallel_tensor_guid_t> result =
          get_incoming_weights(pcg, input_added.parallel_layer);
      std::vector<parallel_tensor_guid_t> correct = {};

      CHECK(result == correct);
    }

    SUBCASE("layer has inputs but no weights") {
      ParallelLayerAddedResult input_added =
          pcg_add_input_layer(pcg, input_shape);
      parallel_tensor_guid_t t_input = get_only(input_added.outputs);

      ParallelLayerAddedResult relu_added = add_parallel_layer(
          pcg, make_layer_attrs(make_relu_attrs()), {t_input}, {});

      std::vector<parallel_tensor_guid_t> result =
          get_incoming_weights(pcg, relu_added.parallel_layer);
      std::vector<parallel_tensor_guid_t> correct = {};

      CHECK(result == correct);
    }

    SUBCASE("layer has inputs and weights, and weights are separate by "
            "parallel ops") {
      std::string my_op_name = "my op";

      ParallelComputationGraph pcg = empty_parallel_computation_graph();

      LinearAttrs linear_attrs = LinearAttrs{
          /*out_channels=*/14_n,
          /*use_bias=*/false,
          /*data_type=*/DataType::FLOAT,
          /*activation=*/Activation::RELU,
          /*regularizer=*/std::nullopt,
      };

      ParallelLayerAddedResult input_added =
          pcg_add_input_layer(pcg, input_shape);
      parallel_tensor_guid_t t_input = get_only(input_added.outputs);

      RepartitionAttrs partition_input_attrs = RepartitionAttrs{
          /*repartition_dim=*/ff_dim_t{0_n},
          /*repartition_degree=*/2_n,
      };

      ParallelLayerAddedResult partition_input_added = add_parallel_layer(
          pcg, make_layer_attrs(partition_input_attrs), {t_input}, {});
      parallel_tensor_guid_t t_partitioned_input =
          get_only(partition_input_added.outputs);

      WeightAttrs projection_weight_attrs = WeightAttrs{
          /*tensor_shape=*/throw_if_unexpected(
              get_projection_shape(linear_attrs, input_shape)),
          /*initializer=*/InitializerAttrs{ZeroInitializerAttrs{}},
      };

      ParallelLayerAddedResult projection_weight_added = add_parallel_layer(
          pcg, make_layer_attrs(projection_weight_attrs), {}, {});
      parallel_tensor_guid_t t_projection_weight =
          get_only(projection_weight_added.outputs);

      ReplicateAttrs replicate_projection_attrs = ReplicateAttrs{
          /*replicate_degree=*/2_n,
      };
      ParallelLayerAddedResult replicate_projection_added =
          add_parallel_layer(pcg,
                             make_layer_attrs(replicate_projection_attrs),
                             {t_projection_weight},
                             {});
      parallel_tensor_guid_t t_replicated_projection_weight =
          get_only(replicate_projection_added.outputs);

      ParallelLayerAddedResult linear_added =
          add_parallel_layer(pcg,
                             make_layer_attrs(linear_attrs),
                             {t_partitioned_input},
                             {t_replicated_projection_weight});

      std::vector<parallel_tensor_guid_t> result =
          get_incoming_weights(pcg, linear_added.parallel_layer);
      std::vector<parallel_tensor_guid_t> correct = {
          t_replicated_projection_weight};

      CHECK(result == correct);
    }
  }

  TEST_CASE("pcg_add_input_layer") {
    TensorShape input_shape = TensorShape{
        TensorDims{
            FFOrdered<nonnegative_int>{
                12_n,
                10_n,
            },
        },
        DataType::FLOAT,
    };

    ParallelComputationGraph result = [&] {
      ParallelComputationGraph pcg = empty_parallel_computation_graph();
      pcg_add_input_layer(pcg, input_shape);
      return pcg;
    }();

    ParallelComputationGraph correct = [&] {
      ParallelComputationGraph pcg = empty_parallel_computation_graph();

      ParallelLayerAttrs layer_attrs = ParallelLayerAttrs{
          /*op_attrs=*/PCGOperatorAttrs{InputAttrs{input_shape}},
          /*name=*/std::nullopt,
      };

      add_parallel_layer(/*pcg=*/pcg,
                         /*layer_attrs=*/layer_attrs,
                         /*inputs=*/{},
                         /*weights=*/{},
                         /*output_labels=*/std::vector{CreateGrad::NO});

      return pcg;
    }();

    CHECK(pcgs_are_isomorphic(result, correct));
  }
}
