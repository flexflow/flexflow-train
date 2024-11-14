#include "models/simvp/simvp.h"

namespace FlexFlow {

SimVPConfig get_default_simvp_config() {
  return SimVPConfig{
      /*batch_size=*/1,
      /*hid_S=*/16,
      /*hid_T=*/256,
      /*N_S=*/4,
      /*N_T=*/4,
      /*model_type=*/"gSTA",
      /*mlp_ratio=*/8.0,
      /*drop=*/0.0,
      /*drop_path=*/0.0,
      /*spatio_kernel_enc=*/3,
      /*spatio_kernel_dec=*/3,
      /*in_shape=*/
      {10, 3, 32, 32},
  };
}

// TODO: improve with a transform call
std::vector<bool> create_simvp_samplings(size_t N_S, bool reverse) {
  std::vector<bool> samplings(N_S, false);
  for (size_t i = 0; i < N_S; i++) {
    if (i % 2 == 1) {
      samplings[i] = true;
    }
  }
  return samplings;
}

// TODO
tensor_guid_t create_simvp_convsc(ComputationGraphBuilder &cgb,
                                  SimVPConfig const &config,
                                  size_t in_dim,
                                  size_t out_dim,
                                  bool downsampling,
                                  tensor_guid_t const &input) {
  return input;
}

std::pair<tensor_guid_t, tensor_guid_t>
    create_simvp_encoder(ComputationGraphBuilder &cgb,
                         SimVPConfig const &config,
                         tensor_guid_t const &input) {
  size_t C = config.in_shape.at(1); // Channel
  std::vector<bool> samplings = create_simvp_samplings(config.N_S);

  tensor_guid_t enc1 =
      create_simvp_convsc(cgb, config, C, config.hid_S, samplings[0], input);
  tensor_guid_t latent = enc1;

  for (size_t i = 1; i < config.N_S; i++) {
    latent = create_simvp_convsc(
        cgb, config, config.hid_S, config.hid_S, samplings[i], latent);
  }

  return {latent, enc1};
}

// TODO
tensor_guid_t create_simvp_middle_net(ComputationGraphBuilder &cgb,
                                      SimVPConfig const &config,
                                      tensor_guid_t const &embed) {
  return embed;
}

// TODO
tensor_guid_t create_simvp_decoder(ComputationGraphBuilder &cgb,
                                   SimVPConfig const &config,
                                   tensor_guid_t const &hid,
                                   tensor_guid_t const &skip) {
  return hid;
}

ComputationGraph get_simvp_computation_graph(SimVPConfig const &config) {
  ComputationGraphBuilder cgb;

  // Create input tensor
  size_t B = config.batch_size;     // Number of samples
  size_t T = config.in_shape.at(0); // Number of frames in each sample
  size_t C = config.in_shape.at(1); // Channel
  size_t H = config.in_shape.at(2); // Height
  size_t W = config.in_shape.at(3); // Width

  TensorShape input_shape = TensorShape{
      TensorDims{FFOrdered<size_t>{B * T, C, H, W}},
      DataType::FLOAT,
  };
  tensor_guid_t input = cgb.create_input(input_shape, CreateGrad::YES);

  // Create the model
  auto [embed, skip] = create_simvp_encoder(cgb, config, input);

  tensor_guid_t hid = create_simvp_middle_net(cgb, config, embed);

  tensor_guid_t output = create_simvp_decoder(cgb, config, hid, skip);

  return cgb.computation_graph;
}

} // namespace FlexFlow
