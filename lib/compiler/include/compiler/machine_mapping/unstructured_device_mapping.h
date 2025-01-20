#ifndef _FLEXFLOW_COMPILER_MACHINE_MAPPING_UNSTRUCTURED_DEVICE_MAPPING_H
#define _FLEXFLOW_COMPILER_MACHINE_MAPPING_UNSTRUCTURED_DEVICE_MAPPING_H

#include "compiler/machine_mapping/machine_mapping.dtg.h"
#include "compiler/machine_mapping/unstructured_device_mapping.dtg.h"
#include "pcg/machine_specification.dtg.h"
#include "pcg/parallel_computation_graph/parallel_computation_graph.dtg.h"

namespace FlexFlow {

UnstructuredDeviceMapping
    get_unstructured_device_mapping(MachineMapping const &machine_mapping,
                                    MachineSpecification const &machine_spec,
                                    ParallelComputationGraph const &pcg);

} // namespace FlexFlow

#endif
