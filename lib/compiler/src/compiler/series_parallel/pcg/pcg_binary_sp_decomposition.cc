#include "compiler/series_parallel/pcg/pcg_binary_sp_decomposition.h"
#include "compiler/series_parallel/pcg/get_pcg_series_parallel_decomposition.h"
#include "compiler/series_parallel/pcg/pcg_binary_series_split.h"
#include "utils/graph/series_parallel/binary_sp_decomposition_tree/generic_binary_sp_decomposition_tree/find_paths_to_leaf.h"
#include "utils/graph/series_parallel/binary_sp_decomposition_tree/generic_binary_sp_decomposition_tree/get_leaves.h"
#include "utils/graph/series_parallel/binary_sp_decomposition_tree/generic_binary_sp_decomposition_tree/get_path_to_leaf_map.h"
#include "utils/graph/series_parallel/series_parallel_decomposition.dtg.h"
#include "utils/overload.h"
#include "compiler/series_parallel/pcg/pcg_binary_parallel_split.h"

namespace FlexFlow {

GenericBinarySPDecompositionTreeImplementation<PCGBinarySPDecomposition,
                                               PCGBinarySeriesSplit,
                                               PCGBinaryParallelSplit,
                                               parallel_layer_guid_t>
    generic_impl_for_pcg_sp_tree() {

  return GenericBinarySPDecompositionTreeImplementation<
      PCGBinarySPDecomposition,
      PCGBinarySeriesSplit,
      PCGBinaryParallelSplit,
      parallel_layer_guid_t>{
      /*series_get_left_child=*/[](PCGBinarySeriesSplit const &split)
                                    -> PCGBinarySPDecomposition const & {
        return split.get_left_child();
      },
      /*parallel_get_left_child=*/
      [](PCGBinaryParallelSplit const &split)
          -> PCGBinarySPDecomposition const & {
        return split.get_left_child();
      },
      /*series_get_right_child=*/
      [](PCGBinarySeriesSplit const &split)
          -> PCGBinarySPDecomposition const & {
        return split.get_right_child();
      },
      /*parallel_get_right_child=*/
      [](PCGBinaryParallelSplit const &split)
          -> PCGBinarySPDecomposition const & {
        return split.get_right_child();
      },
      /*get_node_type=*/
      [](PCGBinarySPDecomposition const &tree) -> SPDecompositionTreeNodeType {
        return get_node_type(tree);
      },
      /*require_series=*/
      [](PCGBinarySPDecomposition const &tree) -> PCGBinarySeriesSplit const & {
        return tree.get<PCGBinarySeriesSplit>();
      },
      /*require_parallel=*/
      [](PCGBinarySPDecomposition const &tree)
          -> PCGBinaryParallelSplit const & {
        return tree.get<PCGBinaryParallelSplit>();
      },
      /*require_leaf=*/
      [](PCGBinarySPDecomposition const &tree)
          -> parallel_layer_guid_t const & {
        return tree.get<parallel_layer_guid_t>();
      },
  };
}

BinarySPDecompositionTree
    binary_sp_tree_from_pcg_sp_tree(PCGBinarySPDecomposition const &pcg_tree) {
  return pcg_tree.visit<BinarySPDecompositionTree>(overload{
      [](PCGBinarySeriesSplit const &series) -> BinarySPDecompositionTree {
        return BinarySPDecompositionTree{
            binary_series_split_from_pcg_series_split(series),
        };
      },
      [](PCGBinaryParallelSplit const &parallel) -> BinarySPDecompositionTree {
        return BinarySPDecompositionTree{
            binary_parallel_split_from_pcg_parallel_split(parallel),
        };
      },
      [](parallel_layer_guid_t const &layer) -> BinarySPDecompositionTree {
        return BinarySPDecompositionTree{
            layer.raw_graph_node,
        };
      },
  });
}

PCGBinarySeriesSplit pcg_binary_series_split_from_binary_series_split(
    BinarySeriesSplit const &split) {
  return PCGBinarySeriesSplit{
      pcg_binary_sp_decomposition_from_binary_sp_decomposition_tree(
          split.get_left_child()),
      pcg_binary_sp_decomposition_from_binary_sp_decomposition_tree(
          split.get_right_child()),
  };
}

PCGBinaryParallelSplit pcg_binary_parallel_split_from_binary_parallel_split(
    BinaryParallelSplit const &split) {
  return PCGBinaryParallelSplit{
      pcg_binary_sp_decomposition_from_binary_sp_decomposition_tree(
          split.get_left_child()),
      pcg_binary_sp_decomposition_from_binary_sp_decomposition_tree(
          split.get_right_child()),
  };
}

PCGBinarySPDecomposition
    pcg_binary_sp_decomposition_from_binary_sp_decomposition_tree(
        BinarySPDecompositionTree const &sp_tree) {

  return sp_tree.visit<PCGBinarySPDecomposition>(overload{
      [](BinarySeriesSplit const &series) -> PCGBinarySPDecomposition {
        return PCGBinarySPDecomposition{
            pcg_binary_series_split_from_binary_series_split(series),
        };
      },
      [](BinaryParallelSplit const &parallel) -> PCGBinarySPDecomposition {
        return PCGBinarySPDecomposition{
            PCGBinaryParallelSplit{
                pcg_binary_sp_decomposition_from_binary_sp_decomposition_tree(
                    parallel.get_left_child()),
                pcg_binary_sp_decomposition_from_binary_sp_decomposition_tree(
                    parallel.get_right_child()),
            },
        };
      },
      [](Node const &node) -> PCGBinarySPDecomposition {
        return PCGBinarySPDecomposition{
            parallel_layer_guid_t{node},
        };
      },
  });
}

std::unordered_multiset<parallel_layer_guid_t>
    get_parallel_layers(PCGBinarySPDecomposition const &tree) {
  return get_leaves(tree, generic_impl_for_pcg_sp_tree());
}

SPDecompositionTreeNodeType
    get_node_type(PCGBinarySPDecomposition const &tree) {
  return tree.visit<SPDecompositionTreeNodeType>(overload{
      [](PCGBinarySeriesSplit const &) {
        return SPDecompositionTreeNodeType::SERIES;
      },
      [](PCGBinaryParallelSplit const &) {
        return SPDecompositionTreeNodeType::PARALLEL;
      },
      [](parallel_layer_guid_t const &) {
        return SPDecompositionTreeNodeType::NODE;
      },
  });
}

std::unordered_set<BinaryTreePath>
    pcg_sp_tree_get_all_leaf_paths(PCGBinarySPDecomposition const &tree) {
  return keys(pcg_sp_tree_get_path_to_leaf_map(tree));
}

std::unordered_set<BinaryTreePath>
    find_paths_to_leaf(PCGBinarySPDecomposition const &tree,
                       parallel_layer_guid_t const &leaf) {
  return find_paths_to_leaf(tree, generic_impl_for_pcg_sp_tree(), leaf);
}

std::unordered_map<BinaryTreePath, parallel_layer_guid_t>
    pcg_sp_tree_get_path_to_leaf_map(PCGBinarySPDecomposition const &tree) {
  return get_path_to_leaf_map(tree, generic_impl_for_pcg_sp_tree());
}

} // namespace FlexFlow
