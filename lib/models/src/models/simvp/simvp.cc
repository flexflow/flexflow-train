#include "models/simvp/simvp.h"

namespace FlexFlow {

SimVPConfig get_default_simvp_config() {
  return SimVPConfig{64};
}

ComputationGraph get_simvp_computation_graph(SimVPConfig const &config) {
  ComputationGraphBuilder cgb;
  return cgb.computation_graph;
}

} // namespace FlexFlow
