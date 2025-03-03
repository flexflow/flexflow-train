/**
 * @file dlrm.h
 *
 * @brief DLRM model
 *
 * @details The DLRM implementation refers to the examples at
 * https://github.com/flexflow/FlexFlow/blob/78307b0e8beb5d41ee003be8b5db168c2b3ef4e2/examples/cpp/DLRM/dlrm.cc
 * and
 * https://github.com/pytorch/torchrec/blob/7e7819e284398d7dc420e3bf149107ad310fa861/torchrec/models/dlrm.py#L440.
 */

#ifndef _FLEXFLOW_LIB_MODELS_INCLUDE_MODELS_DLRM_H
#define _FLEXFLOW_LIB_MODELS_INCLUDE_MODELS_DLRM_H

#include "models/dlrm/dlrm_arch_interaction_op.dtg.h"
#include "models/dlrm/dlrm_config.dtg.h"
#include "pcg/computation_graph_builder.h"

namespace FlexFlow {

// Helper functions to construct the DLRM model

/**
 * @brief Get the default DLRM config.
 *
 * @details The configs here refer to the example at
 * https://github.com/flexflow/FlexFlow/blob/78307b0e8beb5d41ee003be8b5db168c2b3ef4e2/examples/cpp/DLRM/dlrm.cc.
 */
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
