#include "compiler/machine_mapping/machine_mapping.h"
#include "compiler/machine_mapping/machine_view.h"
#include "op-attrs/computation_graph_op_attrs.h"
#include "utils/containers/are_disjoint.h"
#include "utils/containers/binary_merge_disjoint_maps.h"
#include "utils/containers/keys.h"

namespace FlexFlow {

MappedParallelComputationGraph
    mapped_pcg_from_pcg_and_mapping(ParallelComputationGraph const &pcg,
                                    MachineMapping const &mapping) {

  std::unordered_set<parallel_layer_guid_t> pcg_layers =
      get_parallel_layers(pcg);
  std::unordered_set<parallel_layer_guid_t> mapped_layers =
      keys(mapping.machine_views);
  ASSERT(pcg_layers == mapped_layers);

  return MappedParallelComputationGraph{
      /*pcg=*/pcg,
      /*mapped_tasks=*/
      generate_map(
          get_parallel_layers(pcg),
          [&](parallel_layer_guid_t l) -> MappedOperatorTaskGroup {
            ComputationGraphOpAttrs op_attrs =
                compgraph_op_attrs_from_pcg_op_attrs(pcg_get_op_attrs(pcg, l))
                    .value();

            std::unordered_map<TensorSlotName, ParallelTensorDimDegrees>
                inputs_dim_degrees = get_incoming_input_degrees(pcg, l);

            ASSERT(contains_key(mapping.machine_views, l));
            MachineView machine_view = mapping.machine_views.at(l);

            return mapped_operator_task_group_from_machine_view(
                op_attrs, inputs_dim_degrees, machine_view);
          }),
  };
}

MachineMapping combine_disjoint_mappings(MachineMapping const &m1,
                                         MachineMapping const &m2) {
  return MachineMapping{
      binary_merge_disjoint_maps(m1.machine_views, m2.machine_views),
  };
}

bool nodes_are_disjoint(MachineMapping const &m1, MachineMapping const &m2) {
  return are_disjoint(keys(m1.machine_views), keys(m2.machine_views));
}

} // namespace FlexFlow
