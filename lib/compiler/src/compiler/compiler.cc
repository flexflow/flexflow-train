#include "compiler/compiler.h"
#include "compiler/unity_algorithm/unity_algorithm.h"

namespace FlexFlow {

SearchResult optimize(ComputationGraph const &computation_graph,
                      MachineSpecification const &machine_specification,
                      CostEstimator const &cost_estimator,
                      SearchAlgorithm search_algorithm,
                      UnitySearchConfig const &search_config,
                      DeviceType device_type) {
  switch (search_algorithm) {
    case SearchAlgorithm::DATA_PARALLEL:
      throw std::runtime_error(
          "Data parallel search algorithm is not implemented yet");
    case SearchAlgorithm::UNITY: {
      ParallelComputationGraph pcg =
          parallel_computation_graph_from_computation_graph(computation_graph);
      std::vector<Substitution> substitutions; // TODO: Implement this
      return graph_optimize(pcg,
                            cost_estimator,
                            machine_specification,
                            substitutions,
                            search_config,
                            device_type);
    }
    default:
      throw std::runtime_error("Unknown search algorithm");
  }
}

} // namespace FlexFlow
