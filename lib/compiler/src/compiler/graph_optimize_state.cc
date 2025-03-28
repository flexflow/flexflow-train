#include "compiler/graph_optimize_state.h"
#include "compiler/graph_optimize_result.h"
#include "pcg/parallel_computation_graph/parallel_tensor_guid_t.h"

namespace FlexFlow {

GraphOptimizeState::GraphOptimizeState(
    GraphOptimizeResult const &graph_optimize_result, float runtime)
    : graph_optimize_result(graph_optimize_result), runtime(runtime) {}

bool GraphOptimizeState::operator==(GraphOptimizeState const &other) const {
  // Note(@wmdi): This is a hack to implement a partially correct homomorphism
  // check. Switch to the homomorphism check used in substitutions right after
  // https://github.com/flexflow/FlexFlow/pull/1471 is merged.
  auto layers1 = topological_ordering(graph_optimize_result.pcg);
  auto layers2 = topological_ordering(other.graph_optimize_result.pcg);
  if (layers1.size() != layers2.size()) {
    return false;
  }
  std::unordered_map<parallel_tensor_guid_t, parallel_tensor_guid_t> mapping;
  for (size_t i = 0; i < layers1.size(); ++i) {
    if (get_parallel_layer_attrs(graph_optimize_result.pcg, layers1[i]) !=
        get_parallel_layer_attrs(other.graph_optimize_result.pcg, layers2[i])) {
      return false;
    }
    auto inputs1 = get_incoming_tensors(graph_optimize_result.pcg, layers1[i]);
    auto inputs2 =
        get_incoming_tensors(other.graph_optimize_result.pcg, layers2[i]);
    if (inputs1.size() != inputs2.size()) {
      return false;
    }
    for (size_t j = 0; j < inputs1.size(); ++j) {
      if (inputs1[j] != mapping.at(inputs2[j])) {
        return false;
      }
    }
    auto outputs1 = get_layer_outputs(graph_optimize_result.pcg, layers1[i]);
    auto outputs2 =
        get_layer_outputs(other.graph_optimize_result.pcg, layers2[i]);
    if (outputs1.size() != outputs2.size()) {
      return false;
    }
    for (size_t j = 0; j < outputs1.size(); ++j) {
      mapping.emplace(outputs2[j], outputs1[j]);
    }
  }
  return true;
}

bool GraphOptimizeState::operator!=(GraphOptimizeState const &other) const {
  return !(*this == other);
}

bool GraphOptimizeState::operator<(GraphOptimizeState const &other) const {
  return runtime < other.runtime;
}

std::string format_as(GraphOptimizeState const &st) {
  return fmt::format("<GraphOptimizeState graph_optimize_result={} runtime={}>",
                     st.graph_optimize_result,
                     st.runtime);
}

std::ostream &operator<<(std::ostream &s, GraphOptimizeState const &st) {
  return (s << fmt::to_string(st));
}

} // namespace FlexFlow

namespace std {

size_t hash<::FlexFlow::GraphOptimizeState>::operator()(
    ::FlexFlow::GraphOptimizeState const &state) const {
  // TODO(@wmdi): Eventually it might be good to use a proper graph hash like
  // https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.graph_hashing.weisfeiler_lehman_graph_hash.html#networkx.algorithms.graph_hashing.weisfeiler_lehman_graph_hash
  size_t seed = 0;
  auto layers = topological_ordering(state.graph_optimize_result.pcg);
  ::FlexFlow::hash_combine(seed, layers.size());
  for (auto layer : layers) {
    ::FlexFlow::hash_combine(
        seed, get_parallel_layer_attrs(state.graph_optimize_result.pcg, layer));
    auto inputs = get_incoming_tensors(state.graph_optimize_result.pcg, layer);
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
