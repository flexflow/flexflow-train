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
    std::unordered_multiset<AbstractedSingleCommunication> const &single_communications) {

  auto make_singleton_map = [](AbstractedSingleCommunication const &c) {
    return std::unordered_map<AbstractedCommunicationEdge, num_bytes_t>{
      {c.edge, c.size},
    };
  };

  return AbstractedTensorSetMovement{
    merge_maps_with(
      transform(vector_of(single_communications), make_singleton_map),
      [](num_bytes_t lhs, num_bytes_t rhs) {
        return lhs + rhs;
      }),
  };
}


TensorSetMovement concretize_abstracted_tensor_set_movement(
    AbstractedTensorSetMovement const &abstracted,
    std::unordered_map<BinaryTreePath, MachineSpaceStencil> const &pre_machine_stencils,
    std::unordered_map<BinaryTreePath, MachineSpaceStencil> const &post_machine_stencils) {

  return TensorSetMovement{
    /*edge_to_size=*/
      map_keys_with_value_merging(abstracted.edge_to_size,
                                  /*key_func=*/[&](AbstractedCommunicationEdge const &k) {
                                    return concretize_abstracted_communication_edge(
                                      /*edge=*/k,
                                      /*src_machine_stencils=*/pre_machine_stencils,
                                      /*dst_machine_stencils=*/post_machine_stencils);
                                  },
                                  /*merge_values=*/[](num_bytes_t lhs, num_bytes_t rhs) {
                                    return lhs + rhs;
                                  }),
  };
}

} // namespace FlexFlow
