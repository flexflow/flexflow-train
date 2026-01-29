#include "compiler/machine_mapping/machine_mapping_problem_tree/machine_mapping_problem_tree.h"
#include "compiler/machine_mapping/abstracted_tensor_set_movement/abstracted_tensor_set_movement.h"
#include "utils/graph/series_parallel/binary_sp_decomposition_tree/generic_binary_sp_decomposition_tree/as_dot.h"
#include "utils/graph/series_parallel/binary_sp_decomposition_tree/generic_binary_sp_decomposition_tree/get_all_leaf_paths.h"
#include "utils/graph/series_parallel/binary_sp_decomposition_tree/generic_binary_sp_decomposition_tree/get_leaves.h"
#include "utils/graph/series_parallel/binary_sp_decomposition_tree/generic_binary_sp_decomposition_tree/get_path_to_leaf_map.h"
#include "utils/graph/series_parallel/binary_sp_decomposition_tree/generic_binary_sp_decomposition_tree/get_subtree_at_path.h"

namespace FlexFlow {

GenericBinarySPDecompositionTreeImplementation<
    MachineMappingProblemTree,
    MMProblemTreeSeriesSplit,
    MMProblemTreeParallelSplit,
    UnmappedRuntimeOnlyOpCostEstimateKey>
    generic_binary_sp_impl_for_mm_problem_tree() {
  return GenericBinarySPDecompositionTreeImplementation<
      MachineMappingProblemTree,
      MMProblemTreeSeriesSplit,
      MMProblemTreeParallelSplit,
      UnmappedRuntimeOnlyOpCostEstimateKey>{
      /*series_get_left_child=*/[](MMProblemTreeSeriesSplit const &split)
                                    -> MachineMappingProblemTree const & {
        return split.get_left_child();
      },
      /*parallel_get_left_child=*/
      [](MMProblemTreeParallelSplit const &split)
          -> MachineMappingProblemTree const & {
        return split.get_left_child();
      },
      /*series_get_right_child=*/
      [](MMProblemTreeSeriesSplit const &split)
          -> MachineMappingProblemTree const & {
        return split.get_right_child();
      },
      /*parallel_get_right_child=*/
      [](MMProblemTreeParallelSplit const &split)
          -> MachineMappingProblemTree const & {
        return split.get_right_child();
      },
      /*get_node_type=*/
      [](MachineMappingProblemTree const &tree) -> SPDecompositionTreeNodeType {
        return get_node_type(tree);
      },
      /*require_series=*/
      [](MachineMappingProblemTree const &tree)
          -> MMProblemTreeSeriesSplit const & {
        return tree.get<MMProblemTreeSeriesSplit>();
      },
      /*require_parallel=*/
      [](MachineMappingProblemTree const &tree)
          -> MMProblemTreeParallelSplit const & {
        return tree.get<MMProblemTreeParallelSplit>();
      },
      /*require_leaf=*/
      [](MachineMappingProblemTree const &tree)
          -> UnmappedRuntimeOnlyOpCostEstimateKey const & {
        return tree.get<UnmappedRuntimeOnlyOpCostEstimateKey>();
      },
  };
}

SPDecompositionTreeNodeType
    get_node_type(MachineMappingProblemTree const &tree) {
  return tree.visit<SPDecompositionTreeNodeType>(overload{
      [](MMProblemTreeSeriesSplit const &) {
        return SPDecompositionTreeNodeType::SERIES;
      },
      [](MMProblemTreeParallelSplit const &) {
        return SPDecompositionTreeNodeType::PARALLEL;
      },
      [](UnmappedRuntimeOnlyOpCostEstimateKey const &) {
        return SPDecompositionTreeNodeType::NODE;
      },
  });
}

std::unordered_multiset<UnmappedRuntimeOnlyOpCostEstimateKey>
    get_leaves(MachineMappingProblemTree const &tree) {
  return get_leaves(tree, generic_binary_sp_impl_for_mm_problem_tree());
}

std::unordered_set<BinaryTreePath>
    get_all_leaf_paths(MachineMappingProblemTree const &tree) {
  return get_all_leaf_paths(tree, generic_binary_sp_impl_for_mm_problem_tree());
}

std::optional<MachineMappingProblemTree>
    mm_problem_tree_get_subtree_at_path(MachineMappingProblemTree const &tree,
                                        BinaryTreePath const &path) {
  return get_subtree_at_path(
      tree, generic_binary_sp_impl_for_mm_problem_tree(), path);
}

std::unordered_map<BinaryTreePath, UnmappedRuntimeOnlyOpCostEstimateKey>
    mm_problem_tree_get_path_to_leaf_map(
        MachineMappingProblemTree const &tree) {
  return get_path_to_leaf_map(tree,
                              generic_binary_sp_impl_for_mm_problem_tree());
}

std::string as_dot(MachineMappingProblemTree const &tree) {
  std::function<std::string(MMProblemTreeSeriesSplit const &)>
      get_series_label =
          [](MMProblemTreeSeriesSplit const &series) -> std::string {
    auto path_as_dot = [](BinaryTreePath const &path) -> std::string {
      return "(" +
             join_strings(path.entries,
                          ", ",
                          [](BinaryTreePathEntry const &entry) -> std::string {
                            if (entry == BinaryTreePathEntry::LEFT_CHILD) {
                              return "l";
                            } else {
                              assert(entry == BinaryTreePathEntry::RIGHT_CHILD);
                              return "r";
                            }
                          }) +
             ")";
    };

    auto path_set_as_dot =
        [&](std::unordered_set<BinaryTreePath> const &path_set) -> std::string {
      return "(" + join_strings(path_set, ", ", path_as_dot) + ")";
    };

    return fmt::format(
        "srcs={} dsts={}",
        path_set_as_dot(get_src_layers(series.tensor_set_movement)),
        path_set_as_dot(get_dst_layers(series.tensor_set_movement)));
  };

  std::function<std::string(MMProblemTreeParallelSplit const &)>
      get_parallel_label =
          [](MMProblemTreeParallelSplit const &parallel) -> std::string {
    return "P";
  };

  std::function<std::string(UnmappedRuntimeOnlyOpCostEstimateKey const &)> get_leaf_label =
      [](UnmappedRuntimeOnlyOpCostEstimateKey const &leaf) -> std::string { return ""; };

  return as_dot(tree,
                generic_binary_sp_impl_for_mm_problem_tree(),
                get_series_label,
                get_parallel_label,
                get_leaf_label);
}

void debug_print_dot(MachineMappingProblemTree const &tree) {
  std::cout << as_dot(tree) << std::endl;
}

} // namespace FlexFlow
