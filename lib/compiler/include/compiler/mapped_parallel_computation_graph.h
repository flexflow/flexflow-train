#ifndef _FLEXFLOW_LIB_COMPILER_INCLUDE_COMPILER_MAPPED_PARALLEL_COMPUTATION_GRAPH_H
#define _FLEXFLOW_LIB_COMPILER_INCLUDE_COMPILER_MAPPED_PARALLEL_COMPUTATION_GRAPH_H

#include "compiler/machine_mapping/machine_mapping.dtg.h"
#include "compiler/mapped_parallel_computation_graph.dtg.h"
#include "op-attrs/computation_graph_op_attrs.dtg.h"
#include "op-attrs/parallel_tensor_space_coordinate.dtg.h"
#include "pcg/machine_view.dtg.h"

namespace FlexFlow {

MappedParallelComputationGraph
  mapped_pcg_from_pcg_and_mapping(
    ParallelComputationGraph const &,
    MachineMapping const &);

bidict<ParallelTensorSpaceCoordinate, MachineSpaceCoordinate> 
  get_tensor_shard_to_device_coord_mapping(ComputationGraphOpAttrs const &,
                                           MachineView const &);


std::string format_as(MappedParallelComputationGraph const &);
std::ostream &operator<<(std::ostream &, MappedParallelComputationGraph const &);

} // namespace FlexFlow

#endif
