#include "compiler/machine_mapping/abstracted_tensor_set_movement/abstracted_single_tensor_movement.h"
#include "compiler/machine_mapping/abstracted_tensor_set_movement/abstracted_single_tensor_communication_edge.h"
#include "utils/containers/filtermap_keys.h"
#include "utils/containers/map_from_pairs.h"
#include "utils/containers/map_keys_with_value_merging.h"
#include "utils/containers/merge_maps_with.h"
#include "utils/containers/require_same.h"
#include "utils/containers/values.h"
#include "utils/containers/transform.h"
#include "utils/containers/require_all_same1.h"

namespace FlexFlow {

std::unordered_set<BinaryTreePath>
    abstracted_single_tensor_movement_get_dst_layers(AbstractedSingleTensorMovement const &m) {
  return transform(keys(m.edge_to_size),
                   [](AbstractedSingleTensorCommunicationEdge const &e) -> BinaryTreePath {
                     return e.dst.operator_tree_path;
                   });
}

AbstractedSingleTensorMovement
  merge_abstracted_single_tensor_movements(
    std::unordered_multiset<AbstractedSingleTensorMovement> const &movements) {

  std::unordered_multiset<BinaryTreePath> src_paths =
    transform(movements, 
              [](AbstractedSingleTensorMovement const &m) {
                return m.src_op_tree_path;
              });

  BinaryTreePath src_op_tree_path = require_all_same1(src_paths);

  return AbstractedSingleTensorMovement{
    /*src_op_tree_path=*/require_all_same1(src_paths),
    /*edge_to_size=*/merge_maps_with(
      transform(vector_of(movements),
                [](AbstractedSingleTensorMovement const &m) {
                  return m.edge_to_size;
                }),
      [](num_bytes_t l, num_bytes_t r) {
        return l + r;
      }),
  };
}

AbstractedSingleTensorMovement
  abstracted_single_tensor_movement_from_communications(
    BinaryTreePath const &src_op_tree_path,
    std::unordered_set<AbstractedSingleTensorCommunication> const &communications) {

  return AbstractedSingleTensorMovement{
    /*src_op_tree_path=*/src_op_tree_path,
    /*edge_to_size=*/
      map_from_pairs(
        transform(communications,
                  [](AbstractedSingleTensorCommunication const &c) {
                    return std::pair{c.edge, c.size};
                  })),
  };
}

TensorSetMovement concretize_abstracted_single_tensor_movement(
    AbstractedSingleTensorMovement const &abstracted,
    std::unordered_map<BinaryTreePath, MachineSpaceStencil> const &pre_machine_stencils,
    std::unordered_map<BinaryTreePath, MachineSpaceStencil> const &post_machine_stencils) {

  MachineSpaceStencil pre_machine_stencil = 
    pre_machine_stencils.at(abstracted.src_op_tree_path);

  std::unordered_map<std::optional<CommunicationEdge>, num_bytes_t>
    communication_edges =
      map_keys_with_value_merging(abstracted.edge_to_size,
                                  /*key_func=*/[&](AbstractedSingleTensorCommunicationEdge const &k) {
                                    return concretize_abstracted_single_tensor_communication_edge(
                                      /*edge=*/k,
                                      /*src_machine_stencils=*/pre_machine_stencil,
                                      /*dst_machine_stencils=*/post_machine_stencils);
                                  },
                                  /*merge_values=*/[](num_bytes_t lhs, num_bytes_t rhs) {
                                    return require_same(lhs, rhs);
                                  });

  return TensorSetMovement{
    /*edge_to_size=*/
      filtermap_keys(communication_edges,
                     [](std::optional<CommunicationEdge> const &e) {
                       return e;
                     }),
  };
}

} // namespace FlexFlow
