#include "compiler/machine_mapping/abstracted_tensor_set_movement/abstracted_tensor_set_movement.h"
#include "compiler/machine_mapping/abstracted_tensor_set_movement/abstracted_communication_edge.h"
#include "compiler/machine_mapping/parallel_layer_guid_oblivious_machine_mapping.h"
#include "utils/containers/flatmap.h"
#include "utils/containers/merge_maps_with.h"
#include "utils/containers/transform.h"
#include "utils/containers/unordered_set_of.h"
#include "utils/hash/unordered_map.h"
#include "utils/containers/map_keys_with_value_merging.h"

namespace FlexFlow {

AbstractedTensorSetMovement empty_abstracted_tensor_set_movement() {
  return AbstractedTensorSetMovement{{}};
}

std::unordered_set<BinaryTreePath>
    get_src_layers(AbstractedTensorSetMovement const &m) {
  return transform(keys(m.edge_to_size), 
                   [](AbstractedCommunicationEdge const &e) -> BinaryTreePath { 
                     return e.src.operator_tree_path; 
                   });
}

std::unordered_set<BinaryTreePath>
    get_dst_layers(AbstractedTensorSetMovement const &m) {
  return transform(keys(m.edge_to_size), 
                   [](AbstractedCommunicationEdge const &e) -> BinaryTreePath { 
                     return e.dst.operator_tree_path; 
                   });
}

AbstractedTensorSetMovement
  abstracted_tensor_set_movement_from_single_communications(
    std::vector<AbstractedSingleCommunication> const &single_communications) {

  auto make_singleton_map = [](AbstractedSingleCommunication const &c) {
    return std::unordered_map<AbstractedCommunicationEdge, num_bytes_t>{
      {c.edge, c.size},
    };
  };

  return AbstractedTensorSetMovement{
    merge_maps_with(
      transform(single_communications, make_singleton_map),
      [](num_bytes_t lhs, num_bytes_t rhs) {
        return lhs + rhs;
      }),
  };
}


TensorSetMovement concretize_abstracted_tensor_set_movement(
    AbstractedTensorSetMovement const &abstracted,
    std::unordered_map<BinaryTreePath, OperatorTaskSpace> const &pre_task_spaces,
    ParallelLayerGuidObliviousMachineMapping const &pre_mapping,
    std::unordered_map<BinaryTreePath, OperatorTaskSpace> const &post_task_spaces,
    ParallelLayerGuidObliviousMachineMapping const &post_mapping) {

  return TensorSetMovement{
    /*edge_to_size=*/
      map_keys_with_value_merging(abstracted.edge_to_size,
                                  /*key_func=*/[&](AbstractedCommunicationEdge const &k) {
                                    return concretize_abstracted_communication_edge(
                                      /*edge=*/k,
                                      /*src_task_spaces=*/pre_task_spaces,
                                      /*src_mapping=*/pre_mapping,
                                      /*dst_task_spaces=*/post_task_spaces,
                                      /*dst_mapping=*/post_mapping);
                                  },
                                  /*merge_values=*/[](num_bytes_t lhs, num_bytes_t rhs) {
                                    return lhs + rhs;
                                  }),
  };
}

} // namespace FlexFlow
