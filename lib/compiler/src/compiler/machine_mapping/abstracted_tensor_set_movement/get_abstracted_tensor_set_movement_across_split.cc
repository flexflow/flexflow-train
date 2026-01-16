#include "compiler/machine_mapping/abstracted_tensor_set_movement/get_abstracted_tensor_set_movement_across_split.h"
#include "compiler/machine_mapping/abstracted_tensor_set_movement/abstracted_single_communication.dtg.h"
#include "compiler/machine_mapping/abstracted_tensor_set_movement/abstracted_single_tensor_movement.h"
#include "compiler/machine_mapping/abstracted_tensor_set_movement/abstracted_tensor_set_movement.h"
#include "compiler/machine_mapping/transitive_reduced_pcg.h"
#include "compiler/series_parallel/pcg/pcg_binary_sp_decomposition.h"
#include "op-attrs/operator_task_space_to_operator_task_space_mapping.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "op-attrs/tensor_shape.h"
#include "pcg/parallel_computation_graph/parallel_computation_graph.h"
#include "pcg/parallel_computation_graph/parallel_computation_graph_edge.dtg.h"
#include "pcg/parallel_computation_graph/parallel_computation_graph_edge.h"
#include "utils/bidict/algorithms/unordered_set_of.h"
#include "utils/containers/binary_cartesian_product.h"
#include "utils/containers/flatmap.h"
#include "utils/containers/generate_map.h"
#include "utils/containers/get_only.h"
#include "utils/containers/group_by.h"
#include "utils/containers/map_from_pairs.h"
#include "utils/containers/merge_maps_with.h"
#include "utils/containers/transform.h"
#include "utils/containers/unordered_multiset_of.h"
#include "utils/containers/values.h"
#include "utils/containers/vector_of.h"

namespace FlexFlow {

AbstractedSingleTensorMovement get_abstracted_single_tensor_movement_along_edge(
    ParallelComputationGraph const &pcg,
    ParallelComputationGraphEdge const &edge,
    BinaryTreePath const &src_path,
    BinaryTreePath const &dst_path) {

  parallel_layer_guid_t pcg_src = get_src_layer(edge);
  parallel_layer_guid_t pcg_dst = get_dst_layer(edge);

  parallel_tensor_guid_t parallel_tensor = get_parallel_tensor(edge);
  TensorShape tensor_piece =
      get_piece_shape(get_parallel_tensor_shape(pcg, parallel_tensor));

  OperatorTaskSpaceToOperatorTaskSpaceMapping mapping =
      pcg_get_mapping_along_edge(pcg, edge);

  bidict<TaskSpaceCoordinate, TaskSpaceCoordinate> coord_mapping =
      op_to_op_get_coord_mapping(mapping);

  std::unordered_map<AbstractedSingleTensorCommunicationEdge, num_bytes_t>
      single_comms = map_from_pairs(transform(
          unordered_set_of(coord_mapping),
          [&](std::pair<TaskSpaceCoordinate, TaskSpaceCoordinate> const &
                  src_dst) -> std::pair<AbstractedSingleTensorCommunicationEdge,
                                        num_bytes_t> {
            auto [src_task_coord, dst_task_coord] = src_dst;

            return std::pair{
                AbstractedSingleTensorCommunicationEdge{
                    /*src_coord=*/src_task_coord,
                    /*dst=*/AbstractedDevice{dst_path, dst_task_coord},
                },
                get_size_in_bytes(tensor_piece),
            };
          }));

  return AbstractedSingleTensorMovement{
      /*src_op_tree_path=*/src_path,
      /*edge_to_size=*/single_comms,
  };
}

AbstractedTensorSetMovement get_abstracted_tensor_set_movement_across_split(
    TransitiveReducedPCG const &tr_pcg, PCGBinarySeriesSplit const &split) {

  std::unordered_set<ParallelComputationGraphEdge> edges_across_split =
      pcg_get_transitive_reduced_edges_across_split(tr_pcg, split);

  OneToMany<parallel_tensor_guid_t, ParallelComputationGraphEdge>
      edges_by_tensor = group_by(edges_across_split,
                                 [](ParallelComputationGraphEdge const &e) {
                                   return get_parallel_tensor(e);
                                 });

  auto get_src_layer_path = [&](parallel_layer_guid_t layer) -> BinaryTreePath {
    return get_only(find_paths_to_leaf(split.get_left_child(), layer));
  };

  auto get_dst_layer_path = [&](parallel_layer_guid_t layer) -> BinaryTreePath {
    return get_only(find_paths_to_leaf(split.get_right_child(), layer));
  };

  auto to_abstracted_single_tensor_movement =
      [&](ParallelComputationGraphEdge const &pcg_edge)
      -> AbstractedSingleTensorMovement {
    parallel_layer_guid_t pcg_src = get_src_layer(pcg_edge);
    parallel_layer_guid_t pcg_dst = get_dst_layer(pcg_edge);

    return get_abstracted_single_tensor_movement_along_edge(
        /*pcg=*/tr_pcg.full_pcg,
        /*edge=*/pcg_edge,
        /*src_path=*/get_src_layer_path(pcg_src),
        /*dst_path=*/get_dst_layer_path(pcg_dst));
  };

  return AbstractedTensorSetMovement{
      transform(
          edges_by_tensor.right_groups(),
          [&](std::unordered_set<ParallelComputationGraphEdge> const &edges) {
            return merge_abstracted_single_tensor_movements(
                transform(unordered_multiset_of(edges),
                          to_abstracted_single_tensor_movement));
          }),
  };
}

} // namespace FlexFlow
