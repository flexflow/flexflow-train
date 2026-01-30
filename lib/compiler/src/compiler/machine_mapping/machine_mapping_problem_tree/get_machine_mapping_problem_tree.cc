#include "compiler/machine_mapping/machine_mapping_problem_tree/get_machine_mapping_problem_tree.h"
#include "compiler/machine_mapping/abstracted_tensor_set_movement/abstracted_tensor_set_movement.h"
#include "compiler/machine_mapping/abstracted_tensor_set_movement/get_abstracted_tensor_set_movement_across_split.h"
#include "compiler/machine_mapping/machine_mapping_problem_tree/machine_mapping_problem_tree.h"
#include "compiler/machine_mapping/machine_mapping_problem_tree/unmapped_runtime_only_op_cost_estimate_key.h"
#include "compiler/machine_mapping/transitive_reduced_pcg.h"
#include "compiler/series_parallel/pcg/pcg_binary_sp_decomposition.h"
#include "pcg/parallel_computation_graph/parallel_computation_graph.h"
#include "utils/containers/all_of.h"
#include "utils/overload.h"

namespace FlexFlow {

bool is_valid_machine_mapping_problem_tree(
    MachineMappingProblemTree const &problem_tree) {
  return problem_tree.visit<bool>(overload{
      [&](MMProblemTreeSeriesSplit const &series_split) {
        AbstractedTensorSetMovement tensor_movement =
            series_split.tensor_set_movement;

        auto contains_paths =
            [](MachineMappingProblemTree const &t,
               std::unordered_set<BinaryTreePath> const &paths) {
              return all_of(paths, [&](BinaryTreePath const &p) {
                return mm_problem_tree_get_subtree_at_path(t, p).has_value();
              });
            };

        return contains_paths(series_split.get_left_child(),
                              get_src_layers(tensor_movement)) &&
               contains_paths(series_split.get_right_child(),
                              get_dst_layers(tensor_movement)) &&
               is_valid_machine_mapping_problem_tree(
                   series_split.get_left_child()) &&
               is_valid_machine_mapping_problem_tree(
                   series_split.get_right_child());
      },
      [&](MMProblemTreeParallelSplit const &parallel_split) {
        return is_valid_machine_mapping_problem_tree(
                   parallel_split.get_left_child()) &&
               is_valid_machine_mapping_problem_tree(
                   parallel_split.get_right_child());
      },
      [&](UnmappedRuntimeOnlyOpCostEstimateKey const &leaf) { return true; },
  });
}

MachineMappingProblemTree get_machine_mapping_problem_tree(
    ParallelComputationGraph const &pcg,
    PCGBinarySPDecomposition const &sp_decomposition_tree) {
  TransitiveReducedPCG tr_pcg = pcg_get_transitive_reduction(pcg);

  std::function<MachineMappingProblemTree(PCGBinarySPDecomposition const &)>
      to_problem_tree;

  to_problem_tree =
      [&](PCGBinarySPDecomposition const &sp) -> MachineMappingProblemTree {
    return sp.visit<MachineMappingProblemTree>(overload{
        [&](PCGBinarySeriesSplit const &series) {
          AbstractedTensorSetMovement tensor_movement =
              get_abstracted_tensor_set_movement_across_split(tr_pcg, series);
          MachineMappingProblemTree result = MachineMappingProblemTree{
              MMProblemTreeSeriesSplit{
                  /*tensor_set_movement=*/tensor_movement,
                  /*lhs=*/to_problem_tree(series.get_left_child()),
                  /*rhs=*/to_problem_tree(series.get_right_child()),
              },
          };
          ASSERT(is_valid_machine_mapping_problem_tree(result));
          return result;
        },
        [&](PCGBinaryParallelSplit const &parallel) {
          MachineMappingProblemTree result = MachineMappingProblemTree{
              MMProblemTreeParallelSplit{
                  to_problem_tree(parallel.get_left_child()),
                  to_problem_tree(parallel.get_right_child()),
              },
          };
          ASSERT(is_valid_machine_mapping_problem_tree(result));
          return result;
        },
        [&](parallel_layer_guid_t const &leaf) {
          MachineMappingProblemTree result = MachineMappingProblemTree{
              get_unmapped_runtime_only_op_cost_estimate_key_for_layer(pcg,
                                                                       leaf),
          };
          ASSERT(is_valid_machine_mapping_problem_tree(result));
          return result;
        },
    });
  };

  MachineMappingProblemTree mm_tree = to_problem_tree(sp_decomposition_tree);

  ASSERT(is_valid_machine_mapping_problem_tree(mm_tree));

  return mm_tree;
}

} // namespace FlexFlow
