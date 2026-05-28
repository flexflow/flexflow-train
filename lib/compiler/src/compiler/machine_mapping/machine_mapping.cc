#include "compiler/machine_mapping/machine_mapping.h"
#include "compiler/machine_mapping/machine_view.h"
#include "compiler/series_parallel/pcg/pcg_binary_sp_decomposition.h"
#include "op-attrs/computation_graph_op_attrs.h"
#include "op-attrs/pcg_operator_attrs.h"
#include "pcg/machine_compute_resource_slice.h"
#include "pcg/mapped_parallel_computation_graph/mapped_parallel_computation_graph.h"
#include "utils/bidict/algorithms/bidict_from_map.h"
#include "utils/containers/are_disjoint.h"
#include "utils/containers/binary_merge_disjoint_maps.h"
#include "utils/containers/unordered_keys.h"

namespace FlexFlow {

MappedParallelComputationGraph
    mapped_pcg_from_pcg_and_mapping(ParallelComputationGraph const &pcg,
                                    MachineMapping const &mapping) {

  std::unordered_set<parallel_layer_guid_t> pcg_layers =
      get_parallel_layers(pcg);

  std::unordered_set<parallel_layer_guid_t> mapped_layers =
      unordered_keys(mapping.machine_views);

  ASSERT(mapped_layers == pcg_layers);

  auto mapping_for_layer =
      [&](parallel_layer_guid_t l) -> MappedOperatorTaskGroup {
    ComputationGraphOpAttrs op_attrs = assert_unwrap(
        compgraph_op_attrs_from_pcg_op_attrs(pcg_get_op_attrs(pcg, l)));

    std::unordered_map<TensorSlotName, ParallelTensorDimDegrees>
        inputs_dim_degrees = get_incoming_input_degrees(pcg, l);

    ASSERT(contains_key(mapping.machine_views, l));
    MachineView machine_view = mapping.machine_views.at(l);

    return mapped_operator_task_group_from_machine_view(
        op_attrs, inputs_dim_degrees, machine_view);
  };

  std::unordered_map<parallel_layer_guid_t, MappedOperatorTaskGroup>
      mapped_op_task_groups = generate_unordered_map(mapped_layers, mapping_for_layer);

  return mapped_pcg_from_pcg_and_mapped_op_task_groups(pcg,
                                                       mapped_op_task_groups);
}

MachineMapping combine_disjoint_mappings(MachineMapping const &m1,
                                         MachineMapping const &m2) {
  return MachineMapping{
      binary_merge_disjoint_maps(m1.machine_views, m2.machine_views),
  };
}

bool nodes_are_disjoint(MachineMapping const &m1, MachineMapping const &m2) {
  return are_disjoint(unordered_keys(m1.machine_views), unordered_keys(m2.machine_views));
}

std::optional<MachineMapping> get_machine_mapping_from_machine_mapping_result(
    PCGBinarySPDecomposition const &sp_decomposition,
    MachineMappingResult const &mm_result) {

  FeasibleMachineMappingResult feasible_mapping = ({
    if (is_infeasible(mm_result)) {
      return std::nullopt;
    }

    require_feasible(mm_result);
  });

  bidict<BinaryTreePath, parallel_layer_guid_t> path_to_leaf_map =
      bidict_from_map(pcg_sp_tree_get_path_to_leaf_map(sp_decomposition));

  return MachineMapping{
      map_keys(feasible_mapping.machine_mapping.raw_mapping,
               [&](BinaryTreePath const &p) -> parallel_layer_guid_t {
                 return path_to_leaf_map.at_l(p);
               }),
  };
}

} // namespace FlexFlow
