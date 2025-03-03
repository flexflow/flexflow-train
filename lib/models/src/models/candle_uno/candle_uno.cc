#include "models/candle_uno/candle_uno.h"
#include "op-attrs/initializers/glorot_normal_attrs.dtg.h"
#include "utils/containers/repeat_element.h"

namespace FlexFlow {

CandleUnoConfig get_default_candle_uno_config() {
  return CandleUnoConfig{
      /*batch_size=*/64_n,
      /*dense_layers=*/repeat_element(/*num_times=*/4_n, /*element=*/4192_n),
      /*dense_feature_layers=*/
      repeat_element(/*num_times=*/8_n, /*element=*/4192_n),
      /*feature_shapes=*/
      {
          {"dose", 1_n},
          {"cell.rnaseq", 942_n},
          {"drug.descriptors", 5270_n},
          {"drug.fingerprints", 2048_n},
      },
      /*input_features=*/
      {
          {"dose1", "dose"},
          {"dose2", "dose"},
          {"cell.rnaseq", "cell.rnaseq"},
          {"drug1.descriptors", "drug.descriptors"},
          {"drug1.fingerprints", "drug.fingerprints"},
          {"drug2.descriptors", "drug.descriptors"},
          {"drug2.fingerprints", "drug.fingerprints"},
      },
      /*dropout=*/0.1,
      /*residual=*/false};
}

tensor_guid_t create_candle_uno_feature_model(
    ComputationGraphBuilder &cgb,
    CandleUnoConfig const &config,
    tensor_guid_t const &input,
    InitializerAttrs const &kernel_initializer) {
  tensor_guid_t t = input;
  for (nonnegative_int dense_dim : config.dense_feature_layers) {
    t = cgb.dense(t,
                  dense_dim,
                  Activation::RELU,
                  /*use_bias=*/false,
                  /*data_type=*/DataType::FLOAT,
                  /*kernel_initializer=*/kernel_initializer);
    if (config.dropout > 0) {
      t = cgb.dropout(t, config.dropout);
    }
  }
  return t;
}

ComputationGraph
    get_candle_uno_computation_graph(CandleUnoConfig const &config) {
  ComputationGraphBuilder cgb;
  InitializerAttrs kernel_initializer =
      InitializerAttrs{GlorotNormalAttrs{/*seed=*/0}};

  auto create_input_tensor =
      [&](FFOrdered<nonnegative_int> const &dims) -> tensor_guid_t {
    TensorShape input_shape = TensorShape{
        TensorDims{dims},
        DataType::FLOAT,
    };
    return cgb.create_input(input_shape, CreateGrad::YES);
  };

  std::set<std::string> input_models;
  for (auto const &shape : config.feature_shapes) {
    auto const &type = shape.first;
    if (type.find(".") != std::string::npos) {
      std::string base_type = type.substr(0, type.find("."));
      // The string parsing here is to match with original implementation at
      // https://github.com/ECP-CANDLE/Benchmarks/blob/f6a3da8818308c9edcd9fedbcf831dd5968efcdd/Pilot1/Uno/uno_baseline_keras2.py#L178.
      if (base_type == "cell" || base_type == "drug") {
        input_models.insert(type);
      }
    }
  }

  std::vector<tensor_guid_t> all_inputs;
  std::vector<tensor_guid_t> encoded_inputs;

  for (auto const &input_feature : config.input_features) {
    std::string const &feature_name = input_feature.second;
    nonnegative_int shape = config.feature_shapes.at(feature_name);
    tensor_guid_t input = create_input_tensor({config.batch_size, shape});
    all_inputs.push_back(input);

    if (contains(input_models, feature_name)) {
      encoded_inputs.emplace_back(create_candle_uno_feature_model(
          cgb, config, input, kernel_initializer));
    } else {
      encoded_inputs.emplace_back(input);
    }
  }

  tensor_guid_t output =
      cgb.concat(encoded_inputs, /*axis=*/relative_ff_dim_t{1});
  for (nonnegative_int dense_layer_dim : config.dense_layers) {
    tensor_guid_t residual_input = output;
    output = cgb.dense(output,
                       dense_layer_dim,
                       Activation::RELU,
                       /*use_bias=*/false,
                       /*data_type=*/DataType::FLOAT,
                       /*kernel_initializer=*/kernel_initializer);
    if (config.dropout > 0) {
      output = cgb.dropout(output, config.dropout);
    }
    if (config.residual) {
      output = cgb.add(output, residual_input);
    }
  }
  output = cgb.dense(output,
                     /*outDim=*/1_n,
                     /*activation=*/std::nullopt,
                     /*use_bias=*/false,
                     /*data_type=*/DataType::FLOAT,
                     /*kernel_initializer=*/kernel_initializer);

  return cgb.computation_graph;
}

} // namespace FlexFlow
