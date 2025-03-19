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

// Refer to
// https://github.com/chengtan9907/OpenSTL/blob/b658dab3da427c8750c8595316e7ae9d70b818df/openstl/models/simvp_model.py#L51
std::vector<bool> create_simvp_samplings(size_t N_S, bool reverse = false);

// Refer to
// https://github.com/chengtan9907/OpenSTL/blob/b658dab3da427c8750c8595316e7ae9d70b818df/openstl/modules/simvp_modules.py#L57
tensor_guid_t create_simvp_convsc(ComputationGraphBuilder &cgb,
                                  SimVPConfig const &config,
                                  tensor_guid_t const &input,
                                  size_t in_dim,
                                  size_t out_dim,
                                  int kernel_size = 3,
                                  bool downsampling = false,
                                  bool upsampling = false);

// Refer to
// https://github.com/chengtan9907/OpenSTL/blob/b658dab3da427c8750c8595316e7ae9d70b818df/openstl/models/simvp_model.py#L150
tensor_guid_t create_simvp_gsta_meta_block(ComputationGraphBuilder &cgb,
                                           SimVPConfig const &config,
                                           tensor_guid_t const &input,
                                           int in_channels,
                                           int out_channels,
                                           float mlp_ratio = 8.0,
                                           float drop = 0.0,
                                           float drop_path = 0.0);

// Refer to
// https://github.com/chengtan9907/OpenSTL/blob/b658dab3da427c8750c8595316e7ae9d70b818df/openstl/modules/simvp_modules.py#L181
tensor_guid_t create_simvp_ga_sub_block(ComputationGraphBuilder &cgb,
                                        SimVPConfig const &config,
                                        tensor_guid_t const &input,
                                        int dim,
                                        int kernel_size = 21,
                                        float mlp_ratio = 4.0,
                                        float drop = 0.0,
                                        float drop_path = 0.1,
                                        float init_value = 1e-2);

// Refer to
// https://github.com/chengtan9907/OpenSTL/blob/b658dab3da427c8750c8595316e7ae9d70b818df/openstl/models/simvp_model.py#L57
std::pair<tensor_guid_t, tensor_guid_t>
    create_simvp_encoder(ComputationGraphBuilder &cgb,
                         SimVPConfig const &config,
                         tensor_guid_t const &input);

// Refer to
// https://github.com/chengtan9907/OpenSTL/blob/b658dab3da427c8750c8595316e7ae9d70b818df/openstl/models/simvp_model.py#L100
tensor_guid_t create_simvp_middle_net(ComputationGraphBuilder &cgb,
                                      SimVPConfig const &config,
                                      tensor_guid_t const &embed,
                                      int channel_in,
                                      int channel_hid,
                                      float mlp_ratio = 4.0,
                                      float drop = 0.0,
                                      float drop_path = 0.1);

// Refer to
// https://github.com/chengtan9907/OpenSTL/blob/b658dab3da427c8750c8595316e7ae9d70b818df/openstl/models/simvp_model.py#L78
tensor_guid_t create_simvp_decoder(ComputationGraphBuilder &cgb,
                                   SimVPConfig const &config,
                                   tensor_guid_t const &hid,
                                   tensor_guid_t const &skip);

/**
 * @brief Get the SimVP computation graph.
 *
 * @details Refered OpenSTL implementation at
 * https://github.com/chengtan9907/OpenSTL/blob/b658dab3da427c8750c8595316e7ae9d70b818df/openstl/models/simvp_model.py#L9
 *
 * @param SimVPConfig The config of the SimVP model.
 * @return ComputationGraph The computation graph of a SimVP model.
 */
ComputationGraph get_simvp_computation_graph(SimVPConfig const &config);

} // namespace FlexFlow

#endif
