#include "compiler/machine_mapping/machine_mapping_mutation_set.h"
#include "compiler/machine_mapping/allowed_machine_views.h"
#include "pcg/machine_view.h"
#include "pcg/operator_task_space.h"
#include "utils/containers/vector_of.h"
#include "utils/nonnegative_int/nonnegative_range.h"
#include "utils/random_utils.h"
#include "utils/vector.h"

namespace FlexFlow {

std::optional<MachineMapping>
    get_naive_mapping(ParallelComputationGraph &pcg,
                      MachineSpecification const &resources,
                      DeviceType const &device_type) {
  std::vector<parallel_layer_guid_t> layers = topological_ordering(pcg);
  std::unordered_map<parallel_layer_guid_t, MachineView> machine_views;
  for (parallel_layer_guid_t layer : layers) {
    OperatorTaskSpace task = get_operator_task_space(pcg, layer);
    std::unordered_set<MachineView> allowed_machine_views =
        get_allowed_machine_views(resources, task, DeviceType::GPU);
    if (allowed_machine_views.empty()) {
      return std::nullopt;
    }
    machine_views.insert({layer, *(allowed_machine_views.begin())});
  }
  return MachineMapping{machine_views};
}

std::optional<MachineMapping>
    get_random_mutation(SearchResult mapped_pcg,
                        MachineSpecification const &resources,
                        DeviceType const &device_type) {
  ParallelComputationGraph pcg = mapped_pcg.pcg;
  std::vector<parallel_layer_guid_t> layers = topological_ordering(pcg);
  if (layers.size() == 0) {
    return std::nullopt;
  }
  parallel_layer_guid_t random_layer = select_random(layers);

  MachineMapping machine_mapping = mapped_pcg.machine_mapping;
  MachineView machine_view = machine_mapping.machine_views.at(random_layer);
  OperatorTaskSpace task = get_operator_task_space(pcg, random_layer);

  std::vector<MachineView> allowed_machine_views =
      vector_of(get_allowed_machine_views(resources, task, device_type));
  MachineView random_new_machine_view = select_random(allowed_machine_views);

  machine_mapping.machine_views.at(random_layer) = random_new_machine_view;
  return machine_mapping;
}
} // namespace FlexFlow
