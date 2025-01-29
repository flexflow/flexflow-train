#ifndef _FLEXFLOW_COMPILER_MACHINE_MAPPING_H
#define _FLEXFLOW_COMPILER_MACHINE_MAPPING_H

#include "compiler/machine_mapping/machine_mapping.dtg.h"
#include "compiler/machine_mapping/machine_mapping_result.h"
#include "compiler/series_parallel/pcg/pcg_binary_sp_decomposition.dtg.h"

namespace FlexFlow {

MachineMapping combine_disjoint_mappings(MachineMapping const &,
                                         MachineMapping const &);

bool nodes_are_disjoint(MachineMapping const &m1, MachineMapping const &m2);

parallel_layer_guid_t
    get_layer_from_path(PCGBinarySPDecomposition const &sp_decomposition,
                        BinaryTreePath const &path);

std::optional<MachineMapping> get_machine_mapping_from_machine_mapping_result(
    PCGBinarySPDecomposition const &, MachineMappingResult const &);

} // namespace FlexFlow

#endif
