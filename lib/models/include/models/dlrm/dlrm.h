/**
 * @file dlrm.h
 *
 * @brief DLRM model
 *
 * @details The DLRM implementation refers to the examples at
 * https://github.com/flexflow/FlexFlow/blob/inference/examples/cpp/DLRM/dlrm.cc
 * and
 * https://github.com/pytorch/torchrec/blob/main/torchrec/models/dlrm.py#L440.
 */

#ifndef _FLEXFLOW_LIB_MODELS_INCLUDE_MODELS_DLRM_H
#define _FLEXFLOW_LIB_MODELS_INCLUDE_MODELS_DLRM_H

#include "models/dlrm/dlrm_config.dtg.h"
#include "pcg/computation_graph_builder.h"

namespace FlexFlow {

// Helper functions to construct the DLRM model

DLRMConfig get_default_dlrm_config();

tensor_guid_t create_dlrm_mlp(ComputationGraphBuilder &cgb,
                              DLRMConfig const &config,
                              tensor_guid_t const &input,
                              std::vector<size_t> const &mlp_layers);

tensor_guid_t create_dlrm_sparse_embedding_network(ComputationGraphBuilder &cgb,
                                                   DLRMConfig const &config,
                                                   tensor_guid_t const &input,
                                                   int input_dim,
                                                   int output_dim);

tensor_guid_t create_dlrm_interact_features(
    ComputationGraphBuilder &cgb,
    DLRMConfig const &config,
    tensor_guid_t const &bottom_mlp_output,
    std::vector<tensor_guid_t> const &emb_outputs);

/**
 * @brief Get the DLRM computation graph.
 *
 * @param DLRMConfig The config of DLRM model.
 * @return ComputationGraph The computation graph of a DLRM model.
 */
ComputationGraph get_dlrm_computation_graph(DLRMConfig const &config);

} // namespace FlexFlow

#endif
