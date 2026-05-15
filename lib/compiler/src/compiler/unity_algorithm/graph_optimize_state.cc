#include "compiler/unity_algorithm/graph_optimize_state.h"
#include "compiler/machine_mapping/machine_mapping_result.h"
#include "compiler/machine_mapping/machine_view.h"
#include "compiler/series_parallel/pcg/pcg_binary_sp_decomposition.h"
#include "op-attrs/computation_graph_op_attrs.h"
#include "pcg/parallel_computation_graph/parallel_computation_graph_edge.h"
#include "pcg/parallel_computation_graph/parallel_tensor_guid_t.h"
#include "utils/bidict/algorithms/bidict_from_map.h"
#include "utils/containers/zip_values_strict.h"
#include "utils/containers/zip_values_strict_with.h"
#include "utils/hash/tuple.h"
#include "utils/hash/unordered_map.h"
#include "utils/hash/unordered_multiset.h"

namespace FlexFlow {

GraphOptimizeState::GraphOptimizeState(ParallelComputationGraph const &pcg,
                                       milliseconds_t runtime)
    : pcg(pcg), runtime(runtime) {}

static std::unordered_multiset<std::tuple<
    ParallelLayerAttrs,
    std::unordered_map<
        TensorSlotName,
        std::tuple<ParallelLayerAttrs, TensorSlotName, ParallelTensorAttrs>>,
    std::unordered_map<TensorSlotName, ParallelTensorAttrs>>>
    get_layer_signature_set(ParallelComputationGraph const &pcg) {

  auto get_layer_signature = [&](parallel_layer_guid_t l)
      -> std::tuple<ParallelLayerAttrs,
                    std::unordered_map<TensorSlotName,
                                       std::tuple<ParallelLayerAttrs,
                                                  TensorSlotName,
                                                  ParallelTensorAttrs>>,
                    std::unordered_map<TensorSlotName, ParallelTensorAttrs>> {
    ParallelLayerAttrs layer_attrs = get_parallel_layer_attrs(pcg, l);

    std::unordered_map<
        TensorSlotName,
        std::tuple<ParallelLayerAttrs, TensorSlotName, ParallelTensorAttrs>>
        inputs = map_values(
            get_incoming_tensors(pcg, l), [&](parallel_tensor_guid_t const &i) {
              parallel_layer_guid_t src = get_source_layer(i);
              TensorSlotName src_slot = i.raw_graph_output.slot_name;
              ParallelTensorAttrs tensor_attrs =
                  get_parallel_tensor_attrs(pcg, i);

              return std::tuple{
                  get_parallel_layer_attrs(pcg, src),
                  src_slot,
                  tensor_attrs,
              };
            });

    std::unordered_map<TensorSlotName, ParallelTensorAttrs> outputs =
        map_values(get_outgoing_tensors(pcg, l),
                   [&](parallel_tensor_guid_t const &o) {
                     return get_parallel_tensor_attrs(pcg, o);
                   });

    return {
        layer_attrs,
        inputs,
        outputs,
    };
  };

  return transform(unordered_multiset_of(get_parallel_layers(pcg)),
                   get_layer_signature);
}

bool GraphOptimizeState::operator==(GraphOptimizeState const &other) const {
  return get_layer_signature_set(this->pcg) ==
         get_layer_signature_set(other.pcg);
}

bool GraphOptimizeState::operator!=(GraphOptimizeState const &other) const {
  return !(*this == other);
}

bool GraphOptimizeState::operator<(GraphOptimizeState const &other) const {
  return runtime < other.runtime;
}

std::string format_as(GraphOptimizeState const &s) {
  return fmt::format(
      "<SearchResult\nruntime={}\npcg={}>", s.runtime, pcg_as_dot(s.pcg));
}

std::ostream &operator<<(std::ostream &s, GraphOptimizeState const &x) {
  return (s << fmt::to_string(x));
}

} // namespace FlexFlow

namespace std {

size_t hash<::FlexFlow::GraphOptimizeState>::operator()(
    ::FlexFlow::GraphOptimizeState const &state) const {
  // TODO(@wmdi): Eventually it might be good to use a proper graph hash like
  // https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.graph_hashing.weisfeiler_lehman_graph_hash.html#networkx.algorithms.graph_hashing.weisfeiler_lehman_graph_hash
  using namespace ::FlexFlow;

  auto layers = get_layer_signature_set(state.pcg);

  return get_std_hash(layers);
}

} // namespace std
