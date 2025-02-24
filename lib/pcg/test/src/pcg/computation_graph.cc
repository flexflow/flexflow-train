#include "pcg/computation_graph.h"
#include "op-attrs/ops/linear.h"
#include "pcg/computation_graph_builder.h"
#include "utils/containers/get_only.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("get_incoming_inputs(ComputationGraph, layer_guid_t)") {
    SUBCASE("layer has no inputs") {
      std::string input_name = "input";
      ComputationGraph cg = [&] {
        ComputationGraphBuilder b;

        TensorShape input_shape = TensorShape{
            TensorDims{FFOrdered<nonnegative_int>{
                10_n,
                12_n,
            }},
            DataType::FLOAT,
        };

        tensor_guid_t input =
            b.create_input(input_shape, CreateGrad::YES, input_name);

        return b.computation_graph;
      }();

      layer_guid_t input_layer = get_layer_by_name(cg, input_name);

      std::vector<tensor_guid_t> result = get_incoming_inputs(cg, input_layer);
      std::vector<tensor_guid_t> correct = {};

      CHECK(result == correct);
    }

    SUBCASE("layer has inputs but no weights") {
      std::string layer_name = "my op";

      ComputationGraphBuilder b;

      TensorShape input_shape = TensorShape{
          TensorDims{FFOrdered<nonnegative_int>{
              10_n,
              12_n,
          }},
          DataType::FLOAT,
      };

      tensor_guid_t input = b.create_input(input_shape, CreateGrad::YES);
      b.relu(input, layer_name);

      ComputationGraph cg = b.computation_graph;

      layer_guid_t layer = get_layer_by_name(cg, layer_name);

      std::vector<tensor_guid_t> result = get_incoming_inputs(cg, layer);
      std::vector<tensor_guid_t> correct = {input};

      CHECK(result == correct);
    }

    SUBCASE("layer has inputs and weights") {
      std::string layer_name = "my op";

      ComputationGraphBuilder b;

      TensorShape input_shape = TensorShape{
          TensorDims{FFOrdered<nonnegative_int>{
              10_n,
              12_n,
          }},
          DataType::FLOAT,
      };

      tensor_guid_t input = b.create_input(input_shape, CreateGrad::YES);
      b.dense(input,
              /*outDim=*/14_n,
              /*activation=*/Activation::RELU,
              /*use_bias=*/true,
              /*data_type=*/DataType::FLOAT,
              /*projection_initializer=*/std::nullopt,
              /*bias_initializer=*/std::nullopt,
              /*name=*/layer_name);

      ComputationGraph cg = b.computation_graph;

      layer_guid_t dense_layer = get_layer_by_name(cg, layer_name);

      std::vector<tensor_guid_t> result = get_incoming_inputs(cg, dense_layer);
      std::vector<tensor_guid_t> correct = {
          input,
      };

      CHECK(result == correct);
    }
  }

  TEST_CASE("get_incoming_weights(ComputationGraph, layer_guid_t)") {
    SUBCASE("layer has no inputs or weights") {
      std::string input_name = "input";
      ComputationGraph cg = [&] {
        ComputationGraphBuilder b;

        TensorShape input_shape = TensorShape{
            TensorDims{FFOrdered<nonnegative_int>{
                10_n,
                12_n,
            }},
            DataType::FLOAT,
        };

        tensor_guid_t input =
            b.create_input(input_shape, CreateGrad::YES, input_name);

        return b.computation_graph;
      }();

      layer_guid_t input_layer = get_layer_by_name(cg, input_name);

      std::vector<tensor_guid_t> result = get_incoming_weights(cg, input_layer);
      std::vector<tensor_guid_t> correct = {};

      CHECK(result == correct);
    }

    SUBCASE("layer has inputs but no weights") {
      std::string layer_name = "my op";

      ComputationGraph cg = [&] {
        ComputationGraphBuilder b;

        TensorShape input_shape = TensorShape{
            TensorDims{FFOrdered<nonnegative_int>{
                10_n,
                12_n,
            }},
            DataType::FLOAT,
        };

        tensor_guid_t input = b.create_input(input_shape, CreateGrad::YES);
        b.relu(input, layer_name);

        return b.computation_graph;
      }();

      layer_guid_t layer = get_layer_by_name(cg, layer_name);

      std::vector<tensor_guid_t> result = get_incoming_weights(cg, layer);
      std::vector<tensor_guid_t> correct = {};

      CHECK(result == correct);
    }

    SUBCASE("layer has inputs and weights") {
      ComputationGraph cg = make_empty_computation_graph();

      TensorShape input_shape = TensorShape{
          TensorDims{FFOrdered<nonnegative_int>{
              10_n,
              12_n,
          }},
          DataType::FLOAT,
      };

      auto make_layer_attrs = [](auto const &op_attrs) {
        return LayerAttrs{
            /*op_attrs=*/ComputationGraphOpAttrs{op_attrs},
            /*name=*/std::nullopt,
        };
      };

      LinearAttrs linear_attrs = LinearAttrs{
          /*out_channels=*/14_n,
          /*use_bias=*/true,
          /*data_type=*/DataType::FLOAT,
          /*activation=*/Activation::RELU,
          /*regularizer=*/std::nullopt,
      };

      InitializerAttrs zero_init = InitializerAttrs{ZeroInitializerAttrs{}};

      WeightAttrs projection_weight_attrs = WeightAttrs{
          /*tensor_shape=*/throw_if_unexpected(
              get_projection_shape(linear_attrs, input_shape)),
          /*initializer=*/zero_init,
      };

      WeightAttrs bias_weight_attrs = WeightAttrs{
          /*tensor_shape=*/throw_if_unexpected(
              get_bias_shape(linear_attrs, input_shape)),
          /*initializer=*/zero_init,
      };

      LayerAddedResult input_added = add_input_layer(cg, input_shape);
      tensor_guid_t t_input = get_only(input_added.outputs);

      LayerAddedResult projection_weight_added =
          add_layer(cg, make_layer_attrs(projection_weight_attrs), {}, {});
      tensor_guid_t t_projection_weight =
          get_only(projection_weight_added.outputs);

      LayerAddedResult bias_weight_added =
          add_layer(cg, make_layer_attrs(bias_weight_attrs), {}, {});
      tensor_guid_t t_bias_weight = get_only(bias_weight_added.outputs);

      LayerAddedResult linear_added =
          add_layer(cg,
                    make_layer_attrs(linear_attrs),
                    {t_input},
                    {t_projection_weight, t_bias_weight});

      std::vector<tensor_guid_t> result =
          get_incoming_weights(cg, linear_added.layer);
      std::vector<tensor_guid_t> correct = {
          t_projection_weight,
          t_bias_weight,
      };

      CHECK(result == correct);
    }
  }
}
