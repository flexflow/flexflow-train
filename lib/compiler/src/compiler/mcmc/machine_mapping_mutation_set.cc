#include "compiler/mcmc/machine_mapping_mutation_set.h"
#include "compiler/machine_mapping/allowed_machine_views.h"
#include "pcg/machine_view.h"
#include "pcg/operator_task_space.h"
#include "utils/containers/vector_of.h"
#include "utils/vector.h"

namespace FlexFlow {

bool mutation_is_allowed(ParallelComputationGraph &pcg,
                         parallel_layer_guid_t layer,
                         MachineSpecification const &resources,
                         MachineView machine_view) {
  OperatorTaskSpace task = get_operator_task_space(pcg, layer);
  std::unordered_set<MachineView> allowed_machine_views =
      get_allowed_machine_views(resources, task, DeviceType::GPU);
  return allowed_machine_views.count(machine_view);
}

std::vector<MachineMapping>
    get_possible_mutations(SearchResult mapped_pcg,
                           MachineSpecification const &resources) {
  //each mutation only changes one layer at a time
  ParallelComputationGraph pcg = mapped_pcg.pcg;
  std::vector<parallel_layer_guid_t> layers = topological_ordering(pcg);
  std::vector<MachineMapping> machine_mappings;
  for (parallel_layer_guid_t layer : layers) {
    MachineMapping original_mapping = mapped_pcg.machine_mapping;
    MachineView machine_view = original_mapping.machine_views.at(layer);
    OperatorTaskSpace task = get_operator_task_space(pcg, layer);
    std::vector<MachineView> allowed_machine_views =
        vector_of(get_allowed_machine_views(resources, task, DeviceType::GPU));

    std::vector<MachineMapping> new_machine_mappings =
        transform(allowed_machine_views, [&](MachineView machine_views) {
          MachineMapping original_mapping = mapped_pcg.machine_mapping;
          original_mapping.machine_views.at(layer) = machine_views;
          return original_mapping;
        });
    machine_mappings = concat(machine_mappings, new_machine_mappings);
  }
  return machine_mappings;
}

MachineMapping permute_layers(std::vector<parallel_layer_guid_t> layers,
                              MachineMapping mapping) {
  NOT_IMPLEMENTED();
}

MachineMapping copy_layer(parallel_layer_guid_t source,
                          parallel_layer_guid_t destination,
                          MachineMapping mapping) {
  std::unordered_map<parallel_layer_guid_t, MachineView> machine_views =
      mapping.machine_views;
  MachineView machine_view_to_copy = machine_views.at(source);
  machine_views.try_emplace(destination, machine_view_to_copy);
  return MachineMapping{machine_views};
}

MachineView change_stride(nonnegative_int stride,
                          parallel_layer_guid_t layer,
                          MachineView machine_view,
                          nonnegative_int dim) {
  std::vector<stride_t> strides = get_strides(machine_view);
  strides.at(dim.unwrap_nonnegative()) = stride_t{stride};
  MachineView new_machine_view =
      machine_view_from_strides_and_machine_spec_dimensions(
          machine_view.start, strides, get_dimensions(machine_view));
  return new_machine_view;
}

MachineView change_node_idx(nonnegative_int node_ix,
                            parallel_layer_guid_t layer,
                            MachineView machine_view) {
  MachineView new_machine_view =
      machine_view_from_strides_and_machine_spec_dimensions(
          MachineSpaceCoordinate{node_ix,
                                 machine_view.start.device_idx,
                                 machine_view.start.device_type},
          get_strides(machine_view),
          get_dimensions(machine_view));
  return new_machine_view;
}

MachineView change_device_idx(nonnegative_int device_idx,
                              parallel_layer_guid_t layer,
                              MachineView machine_view) {
  MachineView new_machine_view =
      machine_view_from_strides_and_machine_spec_dimensions(
          MachineSpaceCoordinate{machine_view.start.node_idx,
                                 device_idx,
                                 machine_view.start.device_type},
          get_strides(machine_view),
          get_dimensions(machine_view));
  return new_machine_view;
}

MachineView change_projection(MachineSpecificationDimension projection,
                              parallel_layer_guid_t layer,
                              MachineView machine_view,
                              nonnegative_int dim) {
  std::vector<MachineSpecificationDimension> dims =
      get_dimensions(machine_view);
  dims.at(dim.unwrap_nonnegative()) = projection;
  MachineView new_machine_view =
      machine_view_from_strides_and_machine_spec_dimensions(
          machine_view.start, get_strides(machine_view), dims);
  return new_machine_view;
}
} // namespace FlexFlow
