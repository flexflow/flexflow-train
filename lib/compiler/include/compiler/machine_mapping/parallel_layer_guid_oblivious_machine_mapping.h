#ifndef _FLEXFLOW_LIB_COMPILER_INCLUDE_COMPILER_MACHINE_MAPPING_PARALLEL_LAYER_GUID_OBLIVIOUS_MACHINE_MAPPING_H
#define _FLEXFLOW_LIB_COMPILER_INCLUDE_COMPILER_MACHINE_MAPPING_PARALLEL_LAYER_GUID_OBLIVIOUS_MACHINE_MAPPING_H

#include "compiler/machine_mapping/abstracted_tensor_set_movement/machine_space_stencil.dtg.h"
#include "compiler/machine_mapping/machine_mapping_problem_tree/machine_mapping_problem_tree.dtg.h"
#include "compiler/machine_mapping/parallel_layer_guid_oblivious_machine_mapping.dtg.h"
#include "compiler/series_parallel/pcg/pcg_binary_sp_decomposition.dtg.h"
#include "pcg/parallel_computation_graph/parallel_computation_graph.dtg.h"
#include <optional>

namespace FlexFlow {

ParallelLayerGuidObliviousMachineMapping binary_combine_mappings(
    ParallelLayerGuidObliviousMachineMapping const &pre,
    ParallelLayerGuidObliviousMachineMapping const &post);

ParallelLayerGuidObliviousMachineMapping
    restrict_to_left_child(ParallelLayerGuidObliviousMachineMapping const &);
ParallelLayerGuidObliviousMachineMapping
    restrict_to_right_child(ParallelLayerGuidObliviousMachineMapping const &);

std::optional<MachineView>
    get_machine_view_for_path(ParallelLayerGuidObliviousMachineMapping const &,
                              BinaryTreePath const &);

std::unordered_map<BinaryTreePath, MachineSpaceStencil>
  get_machine_stencils_for_decomposition(ParallelComputationGraph const &pcg,
                                         PCGBinarySPDecomposition const &decomposition,
                                         ParallelLayerGuidObliviousMachineMapping const &mapping);

std::unordered_map<BinaryTreePath, std::optional<MachineSpaceStencil>>
  get_machine_stencils_for_mm_problem_tree(MachineMappingProblemTree const &,
                                           ParallelLayerGuidObliviousMachineMapping const &mapping);

std::unordered_map<BinaryTreePath, MachineSpaceStencil>
  get_machine_stencils_for_partially_mapped_mm_problem_tree(
    MachineMappingProblemTree const &,
    ParallelLayerGuidObliviousMachineMapping const &);

} // namespace FlexFlow

#endif
