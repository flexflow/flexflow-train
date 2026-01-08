#ifndef _FLEXFLOW_LIB_COMPILER_INCLUDE_COMPILER_MACHINE_MAPPING_MACHINE_MAPPING_H
#define _FLEXFLOW_LIB_COMPILER_INCLUDE_COMPILER_MACHINE_MAPPING_MACHINE_MAPPING_H

#include "compiler/machine_mapping/machine_mapping.dtg.h"
#include "pcg/device_id_t.dtg.h"
#include "pcg/machine_specification.dtg.h"
#include "pcg/mapped_parallel_computation_graph/mapped_parallel_computation_graph.dtg.h"
#include "pcg/parallel_computation_graph/parallel_computation_graph.dtg.h"

namespace FlexFlow {

MachineMapping combine_disjoint_mappings(MachineMapping const &,
                                         MachineMapping const &);

bool nodes_are_disjoint(MachineMapping const &m1, MachineMapping const &m2);

MappedParallelComputationGraph
    mapped_pcg_from_pcg_and_mapping(ParallelComputationGraph const &,
                                    MachineMapping const &);

} // namespace FlexFlow

#endif
