#include "compiler/compiler.h"
#include "compiler/unity_algorithm/unity_algorithm.h"
#include "pcg/pcg_from_computation_graph.h"
#include "utils/overload.h"

namespace FlexFlow {

SearchResult optimize(ComputationGraph const &computation_graph,
                      MachineSpecification const &machine_specification,
                      CostEstimator const &cost_estimator,
                      AlgorithmConfig const &search_config) {
  return search_config.visit<SearchResult>(overload{
      [&](DataParallelismConfig const &config) -> SearchResult {
        throw std::runtime_error(
            "Data parallel search algorithm is not implemented yet");
      },
      [&](UnitySearchConfig const &config) {
        ParallelComputationGraph pcg =
            pcg_from_computation_graph(computation_graph);
        std::vector<Substitution> substitutions; // TODO: Implement this
        return graph_optimize(
            pcg, cost_estimator, machine_specification, substitutions, config);
      },
  });
}

} // namespace FlexFlow
