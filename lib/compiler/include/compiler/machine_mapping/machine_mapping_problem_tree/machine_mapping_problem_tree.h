#ifndef _FLEXFLOW_LIB_COMPILER_INCLUDE_COMPILER_MACHINE_MAPPING_MACHINE_MAPPING_PROBLEM_TREE_H
#define _FLEXFLOW_LIB_COMPILER_INCLUDE_COMPILER_MACHINE_MAPPING_MACHINE_MAPPING_PROBLEM_TREE_H

#include "compiler/machine_mapping/abstracted_tensor_set_movement/machine_space_stencil.dtg.h"
#include "compiler/machine_mapping/machine_mapping_problem_tree/machine_mapping_problem_tree.dtg.h"
#include "compiler/machine_mapping/machine_mapping_problem_tree/mm_problem_tree_parallel_split.dtg.h"
#include "compiler/machine_mapping/machine_mapping_problem_tree/mm_problem_tree_series_split.dtg.h"
#include "utils/full_binary_tree/binary_tree_path.dtg.h"
#include "utils/graph/series_parallel/binary_sp_decomposition_tree/generic_binary_sp_decomposition_tree/generic_binary_sp_decomposition_tree_implementation.dtg.h"
#include "utils/graph/series_parallel/sp_decomposition_tree_node_type.dtg.h"

namespace FlexFlow {

GenericBinarySPDecompositionTreeImplementation<
    MachineMappingProblemTree,
    MMProblemTreeSeriesSplit,
    MMProblemTreeParallelSplit,
    UnmappedRuntimeOnlyOpCostEstimateKey>
    generic_binary_sp_impl_for_mm_problem_tree();

SPDecompositionTreeNodeType get_node_type(MachineMappingProblemTree const &);

std::unordered_multiset<UnmappedRuntimeOnlyOpCostEstimateKey>
    get_leaves(MachineMappingProblemTree const &);
std::unordered_set<BinaryTreePath>
    get_all_leaf_paths(MachineMappingProblemTree const &);

std::optional<MachineMappingProblemTree>
    mm_problem_tree_get_subtree_at_path(MachineMappingProblemTree const &,
                                        BinaryTreePath const &);

std::unordered_map<BinaryTreePath, UnmappedRuntimeOnlyOpCostEstimateKey>
    mm_problem_tree_get_path_to_leaf_map(MachineMappingProblemTree const &);

} // namespace FlexFlow

#endif
