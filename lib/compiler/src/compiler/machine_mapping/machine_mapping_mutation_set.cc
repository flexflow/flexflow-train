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
                      MachineSpecification const &resources) {
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
    get_random_mutation_notlazy(SearchResult mapped_pcg,
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

  int mutation_op = select_random(range(6));
  switch (mutation_op) {
    case 0: {
      machine_view = decrement_all_strides(machine_view);
      break;
    }
    case 1: {
      nonnegative_int rand_dim = select_random(
          nonnegative_range(nonnegative_int{num_dims(machine_view)}));
      machine_view = increment_stride(machine_view, rand_dim);
      break;
    }
    case 2: {
      nonnegative_int rand_node_idx =
          select_random(nonnegative_range(resources.num_nodes));
      machine_view = change_node_idx(rand_node_idx, machine_view);
      break;
    }
    case 3: {
      if (device_type == DeviceType::GPU) {
        nonnegative_int rand_device_idx =
            select_random(nonnegative_range(resources.num_gpus_per_node));
        machine_view = change_device_idx(rand_device_idx, machine_view);
      } else {
        nonnegative_int rand_device_idx =
            select_random(nonnegative_range(resources.num_cpus_per_node));
        machine_view = change_device_idx(rand_device_idx, machine_view);
      }
      break;
    }
    case 4: {
      nonnegative_int rand_dim = select_random(
          nonnegative_range(nonnegative_int{num_dims(machine_view)}));
      machine_view = switch_projection(machine_view, rand_dim);
      break;
    }
    case 5: {
      // copy layer
      parallel_layer_guid_t layer_to_copy = select_random(layers);
      machine_view = machine_mapping.machine_views.at(layer_to_copy);
      break;
    }
  }
  OperatorTaskSpace task = get_operator_task_space(pcg, random_layer);
  if (is_valid_machine_view(machine_view, task, resources)) {
    // only apply it if valid
    machine_mapping.machine_views.at(random_layer) = machine_view;
  }
  return machine_mapping;
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
  parallel_layer_guid_t random_layer = layers.at(rand() % layers.size());

  MachineMapping machine_mapping = mapped_pcg.machine_mapping;
  MachineView machine_view = machine_mapping.machine_views.at(random_layer);
  OperatorTaskSpace task = get_operator_task_space(pcg, random_layer);

  std::vector<MachineView> allowed_machine_views =
      vector_of(get_allowed_machine_views(resources, task, DeviceType::GPU));
  MachineView random_new_machine_view =
      allowed_machine_views.at(rand() % allowed_machine_views.size());

  machine_mapping.machine_views.at(random_layer) = random_new_machine_view;
  return machine_mapping;
}

MachineView increment_stride(MachineView machine_view, nonnegative_int dim) {
  std::vector<stride_t> strides = get_strides(machine_view);
  nonnegative_int new_stride =
      strides.at(dim.unwrap_nonnegative()).unwrapped + 1_n;
  return change_stride(new_stride, machine_view, dim);
}

MachineView decrement_all_strides(MachineView machine_view) {
  std::vector<stride_t> strides = get_strides(machine_view);
  for (nonnegative_int dim :
       nonnegative_range(nonnegative_int{num_dims(machine_view)})) {
    nonnegative_int old_stride = strides.at(dim.unwrap_nonnegative()).unwrapped;
    if (old_stride >= 1_n) {
      machine_view =
          change_stride(nonnegative_int{old_stride.unwrap_nonnegative() - 1},
                        machine_view,
                        dim);
    }
  }
  return machine_view;
}

MachineView change_stride(nonnegative_int stride,
                          MachineView machine_view,
                          nonnegative_int dim) {
  std::vector<stride_t> strides = get_strides(machine_view);
  strides.at(dim.unwrap_nonnegative()) = stride_t{stride};
  MachineView new_machine_view =
      machine_view_from_strides_and_machine_spec_dimensions(
          machine_view.start, strides, get_dimensions(machine_view));
  return new_machine_view;
}

MachineView change_node_idx(nonnegative_int node_ix, MachineView machine_view) {
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

MachineView switch_projection(MachineView machine_view, nonnegative_int dim) {
  std::vector<MachineSpecificationDimension> dims =
      get_dimensions(machine_view);
  MachineSpecificationDimension projection = dims.at(dim.unwrap_nonnegative());
  if (projection == MachineSpecificationDimension::INTER_NODE) {
    dims.at(dim.unwrap_nonnegative()) =
        MachineSpecificationDimension::INTRA_NODE;
  } else {
    dims.at(dim.unwrap_nonnegative()) =
        MachineSpecificationDimension::INTER_NODE;
  }
  MachineView new_machine_view =
      machine_view_from_strides_and_machine_spec_dimensions(
          machine_view.start, get_strides(machine_view), dims);
  return new_machine_view;
}
} // namespace FlexFlow
