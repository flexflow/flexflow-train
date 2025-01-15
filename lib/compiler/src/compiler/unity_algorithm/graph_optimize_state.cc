#include "compiler/unity_algorithm/graph_optimize_state.h"
#include "pcg/parallel_computation_graph/parallel_tensor_guid_t.h"

namespace FlexFlow {

GraphOptimizeState::GraphOptimizeState(ParallelComputationGraph const &pcg,
                                       float runtime_with_optimal_mm)
    : pcg(pcg), runtime_with_optimal_mm(runtime_with_optimal_mm) {}

bool GraphOptimizeState::operator==(GraphOptimizeState const &other) const {
  return pcgs_are_isomorphic(pcg, other.pcg);
}

bool GraphOptimizeState::operator!=(GraphOptimizeState const &other) const {
  return !(*this == other);
}

bool GraphOptimizeState::operator<(GraphOptimizeState const &other) const {
  return runtime_with_optimal_mm < other.runtime_with_optimal_mm;
}

} // namespace FlexFlow

namespace std {

size_t hash<::FlexFlow::GraphOptimizeState>::operator()(
    ::FlexFlow::GraphOptimizeState const &state) const {
  // TODO(@wmdi): Eventually it might be good to use a proper graph hash like
  // https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.graph_hashing.weisfeiler_lehman_graph_hash.html#networkx.algorithms.graph_hashing.weisfeiler_lehman_graph_hash
  size_t seed = 0;
  auto layers = topological_ordering(state.pcg);
  ::FlexFlow::hash_combine(seed, layers.size());
  for (auto layer : layers) {
    ::FlexFlow::hash_combine(seed, get_parallel_layer_attrs(state.pcg, layer));
    auto inputs = get_incoming_tensors(state.pcg, layer);
    ::FlexFlow::hash_combine(seed, inputs.size());
    for (auto input : inputs) {
      for (size_t i = 0; i < layers.size(); ++i) {
        if (get_source_layer(input) == layers[i]) {
          ::FlexFlow::hash_combine(seed, i);
          break;
        }
      }
    }
  }
  return seed;
}

} // namespace std
