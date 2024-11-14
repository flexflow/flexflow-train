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
                                  size_t in_dim,
                                  size_t out_dim,
                                  bool downsampling,
                                  tensor_guid_t const &input);

std::pair<tensor_guid_t, tensor_guid_t>
    create_simvp_encoder(ComputationGraphBuilder &cgb,
                         SimVPConfig const &config,
                         tensor_guid_t const &input);

tensor_guid_t create_simvp_middle_net(ComputationGraphBuilder &cgb,
                                      SimVPConfig const &config,
                                      tensor_guid_t const &embed);

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
