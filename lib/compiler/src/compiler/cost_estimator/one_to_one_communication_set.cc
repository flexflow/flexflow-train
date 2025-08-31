#include "compiler/cost_estimator/one_to_one_communication_set.h"
#include "op-attrs/operator_task_space.h"
#include "op-attrs/operator_task_space_to_operator_task_space_mapping.h"
#include "pcg/machine_view.h"
#include "pcg/parallel_computation_graph/parallel_computation_graph.h"
#include "pcg/parallel_computation_graph/parallel_computation_graph_edge.h"
#include "utils/bidict/algorithms/transform_keys.h"
#include "utils/bidict/algorithms/transform_values.h"
#include "utils/orthotope/minimal_dim_domain.h"
#include "utils/bidict/algorithms/exhaustive_relational_join.h"
#include "op-attrs/task_space_coordinate.h"

namespace FlexFlow {

OneToOneCommunicationSet 
  one_to_one_communication_set_from_composition(
    OperatorSpaceToMachineSpaceMapping const &pre_map,
    OperatorTaskSpaceToOperatorTaskSpaceMapping const &op_space_mapping,
    OperatorSpaceToMachineSpaceMapping const &post_map) {

  OperatorTaskSpace l_task_space = 
    operator_task_space_from_minimal_dim_domain(
      require_dim_domain_is_minimal(op_space_mapping.raw_mapping.l_domain));

  OperatorTaskSpace r_task_space = 
    operator_task_space_from_minimal_dim_domain(
      require_dim_domain_is_minimal(op_space_mapping.raw_mapping.r_domain));

  ASSERT(pre_map.operator_task_space == l_task_space);
  ASSERT(post_map.operator_task_space == r_task_space);

  bidict<MachineSpaceCoordinate, TaskSpaceCoordinate>
    raw_pre_map = pre_map.raw_mapping.reversed();

  bidict<TaskSpaceCoordinate, TaskSpaceCoordinate> 
    raw_op_space_mapping = 
      transform_keys(
        transform_values(
          op_space_mapping.raw_mapping.coord_mapping,
          task_space_coordinate_from_dim_coord),
        task_space_coordinate_from_dim_coord);

  bidict<TaskSpaceCoordinate, MachineSpaceCoordinate> 
    raw_post_map = post_map.raw_mapping;

  return OneToOneCommunicationSet{
    /*raw_mapping=*/
      exhaustive_relational_join(
        exhaustive_relational_join(
          raw_pre_map,
          raw_op_space_mapping),
        raw_post_map),
  };
}

OneToOneCommunicationSet get_tensor_set_movement_from_pcg_edge(
    ParallelComputationGraphEdge const &edge,
    ParallelComputationGraph const &pcg,
    MachineComputeSpecification const &machine_compute_specification,
    MachineView const &src_mv,
    MachineView const &dst_mv) {

  parallel_layer_guid_t src_layer = get_src_layer(edge);
  nonnegative_int src_idx = get_src_layer_output_idx(edge);
  parallel_tensor_guid_t tensor = parallel_tensor_guid_t{edge.raw_edge.src};
  parallel_layer_guid_t dst_layer = get_dst_layer(edge);
  nonnegative_int dst_idx = get_dst_layer_input_idx(edge);

  ParallelTensorShape tensor_shape = get_parallel_tensor_shape(pcg, tensor);

  OperatorTaskSpace src_task_space = get_operator_task_space(pcg, src_layer);

  OperatorSpaceToMachineSpaceMapping 
    src_map = get_coordinate_mapping_for_machine_view(
      /*operator_task_space=*/src_task_space,
      /*machine_compute_specification=*/machine_compute_specification,
      /*machine_view=*/src_mv);

  OperatorTaskSpace dst_task_space = get_operator_task_space(pcg, dst_layer);

  OperatorSpaceToMachineSpaceMapping 
    dst_map = get_coordinate_mapping_for_machine_view(
      /*operator_task_space=*/dst_task_space,
      /*machine_compute_specification=*/machine_compute_specification,
      /*machine_view=*/dst_mv);

  OperatorSpaceToParallelTensorSpaceMapping src_to_tensor_mapping = 
    pcg_get_operator_to_output_mappings(pcg, src_layer).at(src_idx.unwrap_nonnegative());

  OperatorSpaceToParallelTensorSpaceMapping dst_to_tensor_mapping = 
    pcg_get_operator_to_output_mappings(pcg, dst_layer).at(dst_idx.unwrap_nonnegative());

  OperatorTaskSpaceToOperatorTaskSpaceMapping 
    src_op_to_dst_op_mapping = 
      op_to_op_mapping_from_composition_through_tensor(
        src_to_tensor_mapping,
        dst_to_tensor_mapping);

  return one_to_one_communication_set_from_composition(
      /*pre_map=*/src_map,
      /*operator_task_space_map=*/src_op_to_dst_op_mapping,
      /*post_map=*/dst_map);
}

} // namespace FlexFlow
