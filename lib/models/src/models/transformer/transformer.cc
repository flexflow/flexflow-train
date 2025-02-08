#include "models/transformer/transformer.h"
#include "pcg/computation_graph.h"

namespace FlexFlow {

TransformerConfig get_default_transformer_config() {
  return TransformerConfig{/*num_features=*/512_n,
                           /*sequence_length=*/512_n,
                           /*batch_size=*/64_n,
                           /*dim_feedforward=*/2048_n,
                           /*num_heads=*/8_n,
                           /*num_encoder_layers=*/6_n,
                           /*num_decoder_layers=*/6_n,
                           /*dropout=*/0.1,
                           /*layer_norm_eps=*/1e-05,
                           /*vocab_size=*/64_n};
}

tensor_guid_t create_feedforward_network(ComputationGraphBuilder &cgb,
                                         TransformerConfig const &config,
                                         tensor_guid_t const &input) {
  tensor_guid_t layer1_out = cgb.dense(
      input, config.dim_feedforward, Activation::RELU, /*use_bias=*/true);
  tensor_guid_t dropout_out = cgb.dropout(layer1_out, config.dropout);
  tensor_guid_t layer2_out = cgb.dense(dropout_out,
                                       config.num_features,
                                       /*activation=*/std::nullopt,
                                       /*use_bias=*/true);
  return cgb.dropout(layer2_out, config.dropout);
};

tensor_guid_t create_transformer_encoder_layer(ComputationGraphBuilder &cgb,
                                               TransformerConfig const &config,
                                               tensor_guid_t const &input) {
  std::vector<relative_ff_dim_t> layer_norm_axis = {
      relative_ff_dim_t{-1}}; // Normalize the last dim
  nonnegative_int kdim = config.dim_feedforward / config.num_heads;
  nonnegative_int vdim = config.dim_feedforward / config.num_heads;
  tensor_guid_t self_attention =
      cgb.multihead_attention(/*query=*/input,
                              /*key=*/input,
                              /*value=*/input,
                              /*embed_dim=*/config.num_features,
                              /*num_heads=*/config.num_heads,
                              /*kdim=*/kdim,
                              /*vdim=*/vdim,
                              /*dropout=*/config.dropout,
                              /*bias=*/false);
  assert(are_tensor_guid_shapes_equivalent(
      cgb.computation_graph, input, self_attention));

  tensor_guid_t normalized = cgb.layer_norm(cgb.add(self_attention, input),
                                            layer_norm_axis,
                                            /*elementwise_affine=*/true,
                                            config.layer_norm_eps);
  assert(are_tensor_guid_shapes_equivalent(
      cgb.computation_graph, input, normalized));

  tensor_guid_t feedforward_output =
      create_feedforward_network(cgb, config, normalized);
  assert(are_tensor_guid_shapes_equivalent(
      cgb.computation_graph, input, feedforward_output));
  return cgb.layer_norm(cgb.add(normalized, feedforward_output),
                        layer_norm_axis,
                        /*elementwise_affine=*/true,
                        config.layer_norm_eps);
}

tensor_guid_t create_transformer_encoder(ComputationGraphBuilder &cgb,
                                         TransformerConfig const &config,
                                         tensor_guid_t const &input) {
  tensor_guid_t t = input;
  for (int i = 0; i < config.num_encoder_layers; i++) {
    t = create_transformer_encoder_layer(cgb, config, t);
  }
  return t;
};

tensor_guid_t
    create_transformer_decoder_layer(ComputationGraphBuilder &cgb,
                                     TransformerConfig const &config,
                                     tensor_guid_t const &input,
                                     tensor_guid_t const &encoder_output) {
  std::vector<relative_ff_dim_t> layer_norm_axis = {
      relative_ff_dim_t{-1}}; // Normalize the last dim
  nonnegative_int kdim = config.dim_feedforward / config.num_heads;
  nonnegative_int vdim = config.dim_feedforward / config.num_heads;
  tensor_guid_t self_attention =
      cgb.multihead_attention(/*query=*/input,
                              /*key=*/input,
                              /*value=*/input,
                              /*embed_dim=*/config.num_features,
                              /*num_heads=*/config.num_heads,
                              /*kdim=*/kdim,
                              /*vdim=*/vdim,
                              /*dropout=*/config.dropout,
                              /*bias=*/false);
  assert(are_tensor_guid_shapes_equivalent(
      cgb.computation_graph, input, self_attention));

  tensor_guid_t self_attention_normalized =
      cgb.layer_norm(cgb.add(input, self_attention),
                     layer_norm_axis,
                     /*elementwise_affine=*/true,
                     config.layer_norm_eps);
  assert(are_tensor_guid_shapes_equivalent(
      cgb.computation_graph, input, self_attention_normalized));

  tensor_guid_t mha =
      cgb.multihead_attention(/*query=*/self_attention_normalized,
                              /*key=*/encoder_output,
                              /*value=*/encoder_output,
                              /*embed_dim=*/config.num_features,
                              /*num_heads=*/config.num_heads,
                              /*kdim=*/kdim,
                              /*vdim=*/vdim,
                              /*dropout=*/config.dropout,
                              /*bias=*/false);
  assert(are_tensor_guid_shapes_equivalent(cgb.computation_graph, input, mha));

  tensor_guid_t mha_normalized =
      cgb.layer_norm(cgb.add(self_attention_normalized, mha),
                     layer_norm_axis,
                     /*elementwise_affine=*/true,
                     config.layer_norm_eps);
  assert(are_tensor_guid_shapes_equivalent(
      cgb.computation_graph, input, mha_normalized));

  tensor_guid_t feedforward_output =
      create_feedforward_network(cgb, config, mha_normalized);
  assert(are_tensor_guid_shapes_equivalent(
      cgb.computation_graph, input, feedforward_output));

  return cgb.layer_norm(cgb.add(mha_normalized, feedforward_output),
                        layer_norm_axis,
                        /*elementwise_affine=*/true,
                        config.layer_norm_eps);
}

tensor_guid_t create_transformer_decoder(ComputationGraphBuilder &cgb,
                                         TransformerConfig const &config,
                                         tensor_guid_t const &input,
                                         tensor_guid_t const &encoder_output) {
  tensor_guid_t t = input;
  for (int i = 0; i < config.num_decoder_layers; i++) {
    t = create_transformer_decoder_layer(cgb, config, t, encoder_output);
  }
  return t;
}

ComputationGraph
    get_transformer_computation_graph(TransformerConfig const &config) {
  ComputationGraphBuilder cgb;

  TensorShape input_shape = TensorShape{
      TensorDims{FFOrdered<nonnegative_int>{
          config.batch_size, config.sequence_length, config.num_features}},
      DataType::FLOAT,
  };
  tensor_guid_t input = cgb.create_input(input_shape, CreateGrad::YES, "input");
  tensor_guid_t target =
      cgb.create_input(input_shape, CreateGrad::YES, "target");

  tensor_guid_t encoder_output = create_transformer_encoder(cgb, config, input);
  tensor_guid_t decoder_output =
      create_transformer_decoder(cgb, config, target, encoder_output);

  tensor_guid_t out_prob = cgb.softmax(cgb.dense(decoder_output,
                                                 /*outDim=*/config.vocab_size,
                                                 Activation::RELU,
                                                 /*use_bias=*/true));
  return cgb.computation_graph;
}

} // namespace FlexFlow
