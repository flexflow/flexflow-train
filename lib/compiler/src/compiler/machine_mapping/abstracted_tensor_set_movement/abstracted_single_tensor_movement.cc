#include "compiler/machine_mapping/abstracted_tensor_set_movement/abstracted_single_tensor_movement.h"
#include "compiler/cost_estimator/one_to_one_communication_set.dtg.h"
#include "compiler/cost_estimator/unresolved_communication_set.h"
#include "compiler/cost_estimator/unstructured_communication_set.dtg.h"
#include "compiler/cost_estimator/unstructured_communication_set.h"
#include "op-attrs/operator_task_space.dtg.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "pcg/machine_view.h"
#include "utils/containers/binary_cartesian_product.h"
#include "utils/containers/cartesian_product.h"
#include "utils/containers/transform.h"

namespace FlexFlow {

SingleTensorMovement 
  concretize_abstracted_single_tensor_movement(
    AbstractedSingleTensorMovement const &abstracted,
    MachineComputeSpecification const &machine_compute_specification,
    ParallelLayerGuidObliviousMachineMapping const &pre_mapping,
    ParallelLayerGuidObliviousMachineMapping const &post_mapping) {

  auto concretize_machine_view_pair = [&](MachineView const &pre_mv, MachineView const &post_mv) 
    -> OneToOneCommunicationSet {
    OperatorTaskSpace pre_task_space = 
      operator_task_space_from_minimal_dim_domain(
        require_dim_domain_is_minimal(abstracted.abstracted_shard_movements.l_domain));

    OperatorSpaceToMachineSpaceMapping pre_map = 
      get_coordinate_mapping_for_machine_view(
        /*operator_task_space=*/pre_task_space,
        /*machine_compute_specification=*/machine_compute_specification,
        /*machine_view=*/pre_mv);

    OperatorTaskSpace post_task_space = 
      operator_task_space_from_minimal_dim_domain(
        require_dim_domain_is_minimal(abstracted.abstracted_shard_movements.l_domain));

    OperatorSpaceToMachineSpaceMapping post_map = 
      get_coordinate_mapping_for_machine_view(
        /*operator_task_space=*/post_task_space,
        /*machine_compute_specification=*/machine_compute_specification,
        /*machine_view=*/post_mv);

    return one_to_one_communication_set_from_composition(
      /*pre_map=*/pre_map,
      /*operator_task_space_map=*/abstracted.abstracted_shard_movements,
      /*post_map=*/post_map);
  };

  std::unordered_set<UnstructuredCommunicationSet>
    unstructured_communication_sets = 
      transform(binary_cartesian_product(abstracted.src_machine_views, abstracted.dst_machine_views),
                [&](std::pair<MachineView, MachineView> const &mvs) -> UnstructuredCommunicationSet {
                  return unstructured_communication_set_from_one_to_one(
                    concretize_machine_view_pair(mvs.first, mvs.second));
                });

  return SingleTensorMovement{
    /*shard_shape=*/get_piece_shape(abstracted.parallel_tensor_shape),
    /*unresolved_communication_set=*/
      unresolved_communication_set_from_communication_sets(unstructured_communication_sets),
  };
}

} // namespace FlexFlow
