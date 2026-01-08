#include "compiler/graph_optimize_state.h"
#include "pcg/parallel_computation_graph/parallel_computation_graph_edge.h"
#include "pcg/parallel_computation_graph/parallel_tensor_guid_t.h"
#include "utils/hash/tuple.h"
#include "utils/hash/unordered_map.h"
#include "utils/hash/unordered_multiset.h"

namespace FlexFlow {

GraphOptimizeState::GraphOptimizeState(
    GraphOptimizeResult const &graph_optimize_result, float runtime)
    : graph_optimize_result(graph_optimize_result), runtime(runtime) {}

static std::unordered_multiset<std::tuple<
    ParallelLayerAttrs,
    MappedOperatorTaskGroup,
    std::unordered_map<
        TensorSlotName,
        std::tuple<ParallelLayerAttrs, TensorSlotName, ParallelTensorAttrs>>,
    std::unordered_map<TensorSlotName, ParallelTensorAttrs>>>
    get_layer_signature_set(MappedParallelComputationGraph const &mapped_pcg) {

  auto get_layer_signature = [&](parallel_layer_guid_t l)
      -> std::tuple<ParallelLayerAttrs,
                    MappedOperatorTaskGroup,
                    std::unordered_map<TensorSlotName,
                                       std::tuple<ParallelLayerAttrs,
                                                  TensorSlotName,
                                                  ParallelTensorAttrs>>,
                    std::unordered_map<TensorSlotName, ParallelTensorAttrs>> {
    ParallelLayerAttrs layer_attrs =
        get_parallel_layer_attrs(mapped_pcg.pcg, l);

    std::unordered_map<
        TensorSlotName,
        std::tuple<ParallelLayerAttrs, TensorSlotName, ParallelTensorAttrs>>
        inputs =
            map_values(get_incoming_tensors(mapped_pcg.pcg, l),
                       [&](parallel_tensor_guid_t const &i) {
                         parallel_layer_guid_t src = get_source_layer(i);
                         TensorSlotName src_slot = i.raw_graph_output.slot_name;
                         ParallelTensorAttrs tensor_attrs =
                             get_parallel_tensor_attrs(mapped_pcg.pcg, i);

                         return std::tuple{
                             get_parallel_layer_attrs(mapped_pcg.pcg, src),
                             src_slot,
                             tensor_attrs,
                         };
                       });

    std::unordered_map<TensorSlotName, ParallelTensorAttrs> outputs =
        map_values(get_layer_outputs(mapped_pcg.pcg, l),
                   [&](parallel_tensor_guid_t const &o) {
                     return get_parallel_tensor_attrs(mapped_pcg.pcg, o);
                   });

    return {
        layer_attrs,
        mapped_pcg.mapped_tasks.at(l),
        inputs,
        outputs,
    };
  };

  return transform(unordered_multiset_of(get_parallel_layers(mapped_pcg.pcg)),
                   get_layer_signature);
}

bool GraphOptimizeState::operator==(GraphOptimizeState const &other) const {
  return get_layer_signature_set(this->graph_optimize_result.mapped_pcg) ==
         get_layer_signature_set(other.graph_optimize_result.mapped_pcg);
}

bool GraphOptimizeState::operator!=(GraphOptimizeState const &other) const {
  return !(*this == other);
}

bool GraphOptimizeState::operator<(GraphOptimizeState const &other) const {
  return runtime < other.runtime;
}

} // namespace FlexFlow

namespace std {

size_t hash<::FlexFlow::GraphOptimizeState>::operator()(
    ::FlexFlow::GraphOptimizeState const &state) const {
  // TODO(@wmdi): Eventually it might be good to use a proper graph hash like
  // https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.graph_hashing.weisfeiler_lehman_graph_hash.html#networkx.algorithms.graph_hashing.weisfeiler_lehman_graph_hash
  using namespace ::FlexFlow;

  auto layers = get_layer_signature_set(state.graph_optimize_result.mapped_pcg);

  return get_std_hash(layers);
}

} // namespace std
