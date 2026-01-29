#ifndef _FLEXFLOW_LIB_COMPILER_INCLUDE_COMPILER_MACHINE_MAPPING_MACHINE_RESOURCE_SPLIT_H
#define _FLEXFLOW_LIB_COMPILER_INCLUDE_COMPILER_MACHINE_MAPPING_MACHINE_RESOURCE_SPLIT_H

#include "pcg/machine_compute_resource_slice.dtg.h"
#include "compiler/machine_mapping/machine_resource_split.dtg.h"
#include "compiler/machine_mapping/machine_view.dtg.h"
#include "compiler/machine_mapping/parallel_layer_guid_oblivious_machine_mapping.dtg.h"

namespace FlexFlow {

std::pair<MachineComputeResourceSlice, MachineComputeResourceSlice>
    apply_resource_split(MachineResourceSplit const &split,
                         MachineComputeResourceSlice const &resources);

std::unordered_set<MachineResourceSplit>
    get_machine_resource_splits(MachineComputeResourceSlice const &);

MachineSpaceCoordinate
    offset_machine_space_coordinate_by(MachineSpaceCoordinate const &,
                                       MachineResourceSplit const &);

MachineView offset_machine_view_by(MachineView const &,
                                   MachineResourceSplit const &);

ParallelLayerGuidObliviousMachineMapping offset_layer_oblivious_mapping_by(
    ParallelLayerGuidObliviousMachineMapping const &mapping,
    MachineResourceSplit const &split);

} // namespace FlexFlow

#endif
