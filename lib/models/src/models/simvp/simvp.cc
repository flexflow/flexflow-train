#include "models/simvp/simvp.h"
#include "models/simvp/simvp_model_type.dtg.h"
#include "utils/containers/range.h"
#include "utils/containers/subvec.h"
#include "utils/containers/transform.h"

namespace FlexFlow {

SimVPConfig get_default_simvp_config() {
  return SimVPConfig{
      /*batch_size=*/1,
      /*hid_S=*/16,
      /*hid_T=*/256,
      /*N_S=*/4,
      /*N_T=*/4,
      /*model_type=*/SimVPModelType::gSTA,
      /*mlp_ratio=*/8.0,
      /*drop=*/0.0,
      /*drop_path=*/0.0,
      /*spatio_kernel_enc=*/3,
      /*spatio_kernel_dec=*/3,
      /*in_shape=*/
      {10, 3, 32, 32},
  };
}

std::vector<bool> create_simvp_samplings(size_t N_S, bool reverse) {
  auto const round_down_to_nearest_even = [](size_t num) -> size_t {
    return (num / 2) * 2;
  };

  auto const change_to_true_at_idx = [&](size_t idx) -> bool {
    return (reverse == false) ? (idx % 2 == 1) : (idx % 2 == 0);
  };

  size_t N_S_even_floor = round_down_to_nearest_even(N_S);

  return transform(range(N_S_even_floor), change_to_true_at_idx);
}

tensor_guid_t create_simvp_convsc(ComputationGraphBuilder &cgb,
                                  SimVPConfig const &config,
                                  tensor_guid_t const &input,
                                  size_t in_dim,
                                  size_t out_dim,
                                  int kernel_size,
                                  bool downsampling,
                                  bool upsampling) {
  int stride = (downsampling == true) ? 2 : 1;
  int padding = (kernel_size - stride + 1) / 2;
  int out_channels = upsampling ? out_dim * 4 : out_dim;

  tensor_guid_t conv_out = cgb.conv2d(input,
                                      out_channels,
                                      kernel_size,
                                      kernel_size,
                                      stride,
                                      stride,
                                      padding,
                                      padding);

  return conv_out;
}

std::pair<tensor_guid_t, tensor_guid_t>
    create_simvp_encoder(ComputationGraphBuilder &cgb,
                         SimVPConfig const &config,
                         tensor_guid_t const &input) {
  size_t C = config.in_shape.at(1); // Channel
  std::vector<bool> samplings = create_simvp_samplings(config.N_S);

  tensor_guid_t enc1 = create_simvp_convsc(cgb,
                                           config,
                                           input,
                                           C,
                                           config.hid_S,
                                           config.spatio_kernel_enc,
                                           samplings[0],
                                           false);
  tensor_guid_t latent = enc1;

  for (size_t i = 1; i < samplings.size(); i++) {
    latent = create_simvp_convsc(cgb,
                                 config,
                                 latent,
                                 config.hid_S,
                                 config.hid_S,
                                 config.spatio_kernel_enc,
                                 samplings.at(i),
                                 false);
  }

  return {latent, enc1};
}

// TODO
tensor_guid_t create_simvp_ga_sub_block(ComputationGraphBuilder &cgb,
                                        SimVPConfig const &config,
                                        tensor_guid_t const &input,
                                        int dim,
                                        int kernel_size,
                                        float mlp_ratio,
                                        float drop,
                                        float drop_path,
                                        float init_value) {
  return input;
}

tensor_guid_t create_simvp_gsta_meta_block(ComputationGraphBuilder &cgb,
                                           SimVPConfig const &config,
                                           tensor_guid_t const &input,
                                           int in_channels,
                                           int out_channels,
                                           float mlp_ratio,
                                           float drop,
                                           float drop_path) {
  tensor_guid_t z = create_simvp_ga_sub_block(/*cgb=*/cgb,
                                              /*config=*/config,
                                              /*input=*/input,
                                              /*dim=*/in_channels,
                                              /*kernel_size=*/21,
                                              /*mlp_ratio=*/mlp_ratio,
                                              /*drop=*/drop,
                                              /*drop_path=*/drop_path);

  if (in_channels == out_channels) {
    return z;
  } else {
    return cgb.conv2d(z, out_channels, 1, 1, 1, 1, 0, 0);
  }
}

tensor_guid_t create_simvp_middle_net(ComputationGraphBuilder &cgb,
                                      SimVPConfig const &config,
                                      tensor_guid_t const &embed,
                                      int channel_in,
                                      int channel_hid,
                                      float mlp_ratio,
                                      float drop,
                                      float drop_path) {
  if (config.model_type != FlexFlow::SimVPModelType::gSTA) {
    throw mk_runtime_error(
        fmt::format("Currently only model_type=gSTA is "
                    "supported, but found model_type={}. "
                    "If you need support for additional "
                    "model_type values, please create an issue.",
                    format_as(config.model_type)));
  }

  tensor_guid_t z = embed;

  // Downsample
  z = create_simvp_gsta_meta_block(/*cgb=*/cgb,
                                   /*config=*/config,
                                   /*input=*/z,
                                   /*in_channels=*/channel_in,
                                   /*out_channels=*/channel_hid,
                                   /*mlp_ratio=*/mlp_ratio,
                                   /*drop=*/drop,
                                   /*drop_path=*/drop_path);

  // Middle layers
  for (int i : range(1, config.N_T - 1)) {
    z = create_simvp_gsta_meta_block(/*cgb=*/cgb,
                                     /*config=*/config,
                                     /*input=*/z,
                                     /*in_channels=*/channel_hid,
                                     /*out_channels=*/channel_hid,
                                     /*mlp_ratio=*/mlp_ratio,
                                     /*drop=*/drop,
                                     /*drop_path=*/drop_path);
  }

  // Upsample
  z = create_simvp_gsta_meta_block(/*cgb=*/cgb,
                                   /*config=*/config,
                                   /*input=*/z,
                                   /*in_channels=*/channel_hid,
                                   /*out_channels=*/channel_in,
                                   /*mlp_ratio=*/mlp_ratio,
                                   /*drop=*/drop,
                                   /*drop_path=*/drop_path);

  return z;
}

tensor_guid_t create_simvp_decoder(ComputationGraphBuilder &cgb,
                                   SimVPConfig const &config,
                                   tensor_guid_t const &hid,
                                   tensor_guid_t const &skip) {

  std::cout << "hid shape: " << cgb.get_shape(hid) << std::endl;
  std::cout << "skip shape: " << cgb.get_shape(skip) << std::endl;

  size_t C = config.in_shape.at(1); // Channel
  std::vector<bool> samplings = create_simvp_samplings(config.N_S, true);

  tensor_guid_t latent = hid;
  for (size_t i = 0; i < samplings.size() - 1; i++) {
    latent = create_simvp_convsc(cgb,
                                 config,
                                 latent,
                                 config.hid_S,
                                 config.hid_S,
                                 config.spatio_kernel_dec,
                                 false,
                                 samplings[i]);
  }

  tensor_guid_t out = create_simvp_convsc(cgb,
                                          config,
                                          cgb.add(latent, skip),
                                          config.hid_S,
                                          config.hid_S,
                                          config.spatio_kernel_dec,
                                          false,
                                          samplings.back());

  return cgb.conv2d(out, C, 1, 1, 1, 1, 0, 0, std::nullopt);
}

ComputationGraph get_simvp_computation_graph(SimVPConfig const &config) {
  ComputationGraphBuilder cgb;

  // Create input tensor
  int B = config.batch_size;        // Number of samples
  size_t T = config.in_shape.at(0); // Number of frames in each sample
  size_t C = config.in_shape.at(1); // Channel
  size_t H = config.in_shape.at(2); // Height
  size_t W = config.in_shape.at(3); // Width

  // std::cout << "B T C H W: " << B << " " << T << " " << C << " " << H << " "
  //           << W << std::endl;

  TensorShape input_shape = TensorShape{
      TensorDims{FFOrdered<size_t>{B * T, C, H, W}},
      DataType::FLOAT,
  };
  tensor_guid_t input = cgb.create_input(input_shape, CreateGrad::YES);

  // Create the model
  auto [embed, skip] = create_simvp_encoder(cgb, config, input);

  // std::cout << "embed shape: " << cgb.get_shape(embed) << std::endl;

  // TODO: need to reshape embed here

  tensor_guid_t hid = create_simvp_middle_net(cgb,
                                              config,
                                              embed,
                                              T * config.hid_S,
                                              config.hid_T,
                                              config.mlp_ratio,
                                              config.drop,
                                              config.drop_path);

  // TODO: need to reshape hid here
  // std::cout << "hid shape: " << cgb.get_shape(hid) << std::endl;
  // auto const embed_shape = cgb.get_shape(embed).dims.ff_ordered;
  // std::vector<int> hid_shape = {
  //     static_cast<int>(B * T),
  //     static_cast<int>(embed_shape.at(ff_dim_t{1})),
  //     static_cast<int>(embed_shape.at(ff_dim_t{2})),
  //     static_cast<int>(embed_shape.at(ff_dim_t{3})),
  // };
  // hid = cgb.reshape(hid, hid_shape);
  // std::cout << "hid new shape: " << cgb.get_shape(hid) << std::endl;

  // TODO: enable below after reshaping
  // tensor_guid_t output = create_simvp_decoder(cgb, config, hid, skip);

  return cgb.computation_graph;
}

} // namespace FlexFlow
