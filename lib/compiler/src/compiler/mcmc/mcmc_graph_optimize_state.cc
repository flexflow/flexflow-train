#include "compiler/mcmc/mcmc_graph_optimize_state.h"
#include "pcg/machine_view.h"
#include "pcg/parallel_computation_graph/parallel_tensor_guid_t.h"

namespace FlexFlow {

MCMCOptimizeState::MCMCOptimizeState(SearchResult const &mapped_pcg,
                                     float runtime)
    : mapped_pcg(mapped_pcg), runtime(runtime) {}

bool MCMCOptimizeState::operator==(MCMCOptimizeState const &other) const {
  return pcgs_are_isomorphic(mapped_pcg.pcg, other.mapped_pcg.pcg) &&
         mapped_pcg.machine_mapping == other.mapped_pcg.machine_mapping &&
         runtime == other.runtime;
}

bool MCMCOptimizeState::operator!=(MCMCOptimizeState const &other) const {
  return !(*this == other);
}

bool MCMCOptimizeState::operator<(MCMCOptimizeState const &other) const {
  return runtime < other.runtime;
}

std::string format_as(MCMCOptimizeState const &r) {
  return fmt::format("<MCMCOptimizeState pcg={} machine_mapping={} runtime={}>",
                     as_dot(r.mapped_pcg.pcg),
                     r.mapped_pcg.machine_mapping,
                     r.runtime);
}

std::ostream &operator<<(std::ostream &s, MCMCOptimizeState const &st) {
  return (s << fmt::to_string(st));
}
} // namespace FlexFlow

namespace std {

size_t hash<::FlexFlow::MCMCOptimizeState>::operator()(
    ::FlexFlow::MCMCOptimizeState const &state) const {
  ::FlexFlow::ParallelComputationGraph pcg = state.mapped_pcg.pcg;
  ::FlexFlow::MachineMapping machine_mapping = state.mapped_pcg.machine_mapping;
  size_t seed = 0;
  ::FlexFlow::hash_combine(seed, state.runtime);
  std::vector<::FlexFlow::parallel_layer_guid_t> layers =
      topological_ordering(pcg);
  ::FlexFlow::hash_combine(seed, layers.size());
  for (::FlexFlow::parallel_layer_guid_t const &layer : layers) {
    ::FlexFlow::hash_combine(seed, get_parallel_layer_attrs(pcg, layer));
    std::vector<::FlexFlow::parallel_tensor_guid_t> inputs =
        get_incoming_tensors(pcg, layer);
    ::FlexFlow::hash_combine(seed, inputs.size());
    for (::FlexFlow::parallel_tensor_guid_t input : inputs) {
      for (size_t i = 0; i < layers.size(); ++i) {
        if (get_source_layer(input) == layers.at(i)) {
          ::FlexFlow::hash_combine(seed, i);
          break;
        }
      }
    }
    ::FlexFlow::MachineView machine_view =
        machine_mapping.machine_views.at(layer);
    ::FlexFlow::hash_combine(seed, machine_view.start.node_idx);
    ::FlexFlow::hash_combine(seed, machine_view.start.device_idx);
    if (get_device_type(machine_view) == ::FlexFlow::DeviceType::CPU) {
      ::FlexFlow::hash_combine(seed, 0);
    } else {
      ::FlexFlow::hash_combine(seed, 1);
    }
    for (::FlexFlow::MachineViewDimension dimension : machine_view.dimensions) {
      ::FlexFlow::hash_combine(seed, dimension.stride.unwrapped);
      if (dimension.projection ==
          ::FlexFlow::MachineSpecificationDimension::INTRA_NODE) {
        ::FlexFlow::hash_combine(seed, 0);
      } else {
        ::FlexFlow::hash_combine(seed, 1);
      }
    }
  }

  return seed;
}

} // namespace std
