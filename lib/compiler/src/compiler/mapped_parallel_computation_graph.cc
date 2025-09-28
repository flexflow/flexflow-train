#include "compiler/mapped_parallel_computation_graph.h"
#include "op-attrs/computation_graph_op_attrs.h"

namespace FlexFlow {

MappedParallelComputationGraph
  mapped_pcg_from_pcg_and_mapping(
    ParallelComputationGraph const &pcg,
    MachineMapping const &mapping) {

  return MappedParallelComputationGraph{
    /*pcg=*/pcg,
    /*mapped_tasks=*/
      generate_map(
        get_parallel_layers(pcg), 
        [&](parallel_layer_guid_t l) -> MappedOperatorTaskGroup {
          ComputationGraphOpAttrs op_attrs = 
            compgraph_op_attrs_from_pcg_op_attrs(pcg_get_op_attrs(pcg, l)).value();

          std::vector<ParallelTensorDimDegrees> inputs_dim_degrees = 
            get_incoming_input_degrees(pcg, l);

          MachineView machine_view = mapping.machine_views.at(l);

          return mapped_operator_task_group_from_machine_view(
            op_attrs, 
            inputs_dim_degrees,
            machine_view);
        }),
  };
}


bidict<ParallelTensorSpaceCoordinate, MachineSpaceCoordinate> 
  get_tensor_shard_to_device_coord_mapping(ComputationGraphOpAttrs const &,
                                           MachineView const &) {
  NOT_IMPLEMENTED(); 
}



std::string format_as(MappedParallelComputationGraph const &mapped_pcg) {
  return fmt::format("<GraphOptimizeResult\npcg={}\nmapped_tasks={}>",
                     as_dot(mapped_pcg.pcg),
                     mapped_pcg.mapped_tasks);
}

std::ostream &operator<<(std::ostream &s, MappedParallelComputationGraph const &mapped_pcg) {
  return (s << fmt::to_string(mapped_pcg));
}

} // namespace FlexFlow
