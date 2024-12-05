#ifndef _FLEXFLOW_LIB_MODELS_INCLUDE_MODELS_SIMVP_H
#define _FLEXFLOW_LIB_MODELS_INCLUDE_MODELS_SIMVP_H

#include "pcg/computation_graph_builder.h"
#include "simvp_config.dtg.h"

namespace FlexFlow {

// Helper functions to construct the SimVP model

/**
 * @brief Get the default configs of SimVP model.
 */
SimVPConfig get_default_simvp_config();

std::vector<bool> create_simvp_samplings(size_t N_S, bool reverse = false);

tensor_guid_t create_simvp_convsc(ComputationGraphBuilder &cgb,
                                  SimVPConfig const &config,
                                  tensor_guid_t const &input,
                                  size_t in_dim,
                                  size_t out_dim,
                                  int kernel_size = 3,
                                  bool downsampling = false,
                                  bool upsampling = false);

tensor_guid_t create_simvp_gsta_meta_block(ComputationGraphBuilder &cgb,
                                           SimVPConfig const &config,
                                           tensor_guid_t const &input,
                                           int in_channels,
                                           int out_channels,
                                           float mlp_ratio = 8.0,
                                           float drop = 0.0,
                                           float drop_path = 0.0);

tensor_guid_t create_simvp_ga_sub_block(ComputationGraphBuilder &cgb,
                                        SimVPConfig const &config,
                                        tensor_guid_t const &input,
                                        int dim,
                                        int kernel_size = 21,
                                        float mlp_ratio = 4.0,
                                        float drop = 0.0,
                                        float drop_path = 0.1,
                                        float init_value = 1e-2);

std::pair<tensor_guid_t, tensor_guid_t>
    create_simvp_encoder(ComputationGraphBuilder &cgb,
                         SimVPConfig const &config,
                         tensor_guid_t const &input);

tensor_guid_t create_simvp_middle_net(ComputationGraphBuilder &cgb,
                                      SimVPConfig const &config,
                                      tensor_guid_t const &embed,
                                      int channel_in,
                                      int channel_hid,
                                      float mlp_ratio = 4.0,
                                      float drop = 0.0,
                                      float drop_path = 0.1);

tensor_guid_t create_simvp_decoder(ComputationGraphBuilder &cgb,
                                   SimVPConfig const &config,
                                   tensor_guid_t const &hid,
                                   tensor_guid_t const &skip);

/**
 * @brief Get the SimVP computation graph.
 *
 * @param SimVPConfig The config of the SimVP model.
 * @return ComputationGraph The computation graph of a SimVP model.
 */
ComputationGraph get_simvp_computation_graph(SimVPConfig const &config);

} // namespace FlexFlow

#endif
