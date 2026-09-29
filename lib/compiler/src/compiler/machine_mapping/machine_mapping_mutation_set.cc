#include "compiler/machine_mapping/machine_mapping_mutation_set.h"
#include "compiler/machine_mapping/allowed_machine_views.h"
#include "compiler/machine_mapping/machine_view.h"
#include "op-attrs/operator_task_space.h"
#include "pcg/machine_compute_resource_slice.h"
#include "utils/containers/vector_of.h"
#include "utils/nonnegative_int/nonnegative_range.h"
#include "utils/random_utils.h"

namespace FlexFlow {

std::optional<MachineMapping>
    get_random_mapping(std::mt19937 &gen,
                       ParallelComputationGraph const &pcg,
                       MachineComputeSpecification const &resources) {
  std::vector<parallel_layer_guid_t> layers = topological_ordering(pcg);
  std::map<parallel_layer_guid_t, MachineView> machine_views;
  for (parallel_layer_guid_t layer : layers) {
    OperatorTaskSpace task = get_operator_task_space(pcg, layer);
    std::set<MachineView> allowed_machine_views = get_allowed_machine_views(
        compute_slice_from_specification(resources), task);
    if (allowed_machine_views.empty()) {
      return std::nullopt;
    }
    machine_views.insert(
        {layer, select_random(gen, vector_of(allowed_machine_views))});
  }
  return MachineMapping{machine_views};
}

std::optional<MachineMapping>
    get_random_mutation(std::mt19937 &gen,
                        SearchResult const &mapped_pcg,
                        MachineComputeSpecification const &resources) {
  ParallelComputationGraph pcg = mapped_pcg.pcg;
  std::vector<parallel_layer_guid_t> layers = topological_ordering(pcg);
  if (layers.size() == 0) {
    return std::nullopt;
  }
  parallel_layer_guid_t random_layer = select_random(gen, layers);

  MachineMapping machine_mapping = mapped_pcg.machine_mapping;
  MachineView machine_view = machine_mapping.machine_views.at(random_layer);
  OperatorTaskSpace task = get_operator_task_space(pcg, random_layer);

  std::vector<MachineView> allowed_machine_views =
      vector_of(get_allowed_machine_views(
          compute_slice_from_specification(resources), task));
  MachineView random_new_machine_view =
      select_random(gen, allowed_machine_views);

  machine_mapping.machine_views.at(random_layer) = random_new_machine_view;
  return machine_mapping;
}
} // namespace FlexFlow
