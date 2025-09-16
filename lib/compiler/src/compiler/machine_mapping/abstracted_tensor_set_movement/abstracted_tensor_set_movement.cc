#include "compiler/machine_mapping/abstracted_tensor_set_movement/abstracted_tensor_set_movement.h"
#include "compiler/cost_estimator/tensor_set_movement.h"
#include "compiler/machine_mapping/abstracted_tensor_set_movement/abstracted_single_tensor_movement.dtg.h"
#include "compiler/machine_mapping/abstracted_tensor_set_movement/abstracted_single_tensor_movement.h"
#include "compiler/machine_mapping/parallel_layer_guid_oblivious_machine_mapping.h"
#include "utils/containers/flatmap.h"
#include "utils/containers/merge_maps_with.h"
#include "utils/containers/transform.h"
#include "utils/containers/unordered_set_of.h"
#include "utils/hash/unordered_map.h"
#include "utils/containers/map_keys_with_value_merging.h"
#include "utils/containers/binary_merge_maps_with.h"

namespace FlexFlow {

AbstractedTensorSetMovement empty_abstracted_tensor_set_movement() {
  return AbstractedTensorSetMovement{{}};
}

AbstractedTensorSetMovement 
  abstracted_tensor_set_movement_from_single_tensor_movement(
    AbstractedSingleTensorMovement const &m) {
  return AbstractedTensorSetMovement{
    /*single_tensor_movements=*/{m},
  };
}

std::unordered_set<BinaryTreePath>
    get_src_layers(AbstractedTensorSetMovement const &m) {
  return transform(m.single_tensor_movements, 
                   [](AbstractedSingleTensorMovement const &e) -> BinaryTreePath { 
                     return e.src_op_tree_path; 
                   });
}

std::unordered_set<BinaryTreePath>
    get_dst_layers(AbstractedTensorSetMovement const &m) {
  return flatmap(m.single_tensor_movements,
                 [](AbstractedSingleTensorMovement const &m) -> std::unordered_set<BinaryTreePath> {
                   return abstracted_single_tensor_movement_get_dst_layers(m);
                 });
}

TensorSetMovement concretize_abstracted_tensor_set_movement(
    AbstractedTensorSetMovement const &abstracted,
    std::unordered_map<BinaryTreePath, MachineSpaceStencil> const &pre_machine_stencils,
    std::unordered_map<BinaryTreePath, MachineSpaceStencil> const &post_machine_stencils) {

  std::vector<TensorSetMovement> single_tensor_movements =
    transform(vector_of(abstracted.single_tensor_movements),
              [&](AbstractedSingleTensorMovement const &m) {
                return concretize_abstracted_single_tensor_movement(
                  m,
                  /*pre_machine_stencils=*/pre_machine_stencils,
                  /*post_machine_stencils=*/post_machine_stencils);
              });

  auto merge_tensor_set_movements = 
    [](TensorSetMovement const &lhs,
       TensorSetMovement const &rhs) -> TensorSetMovement {
    return TensorSetMovement{
      binary_merge_maps_with(
        lhs.edge_to_size,
        rhs.edge_to_size,
        [](num_bytes_t l, num_bytes_t r) {
          return l + r; 
        }),
    };
  };

  return foldl(single_tensor_movements, empty_tensor_set_movement(), merge_tensor_set_movements);
}

} // namespace FlexFlow
