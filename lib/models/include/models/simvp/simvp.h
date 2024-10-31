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

/**
 * @brief Get the SimVP computation graph.
 *
 * @param SimVPConfig The config of the SimVP model.
 * @return ComputationGraph The computation graph of a SimVP model.
 */
ComputationGraph get_simvp_computation_graph(SimVPConfig const &config);

} // namespace FlexFlow

#endif
