#include "compiler/mapped_parallel_computation_graph.h"

namespace FlexFlow {

bidict<ParallelTensorSpaceCoordinate, MachineSpaceCoordinate> 
  get_tensor_shard_to_device_coord_mapping(ComputationGraphOpAttrs const &,
                                           MachineView const &) {
  NOT_IMPLEMENTED(); 
}



std::string format_as(MappedParallelComputationGraph const &mapped_pcg) {
  return fmt::format("<GraphOptimizeResult\npcg={}\nmachine_mapping={}>",
                     as_dot(mapped_pcg.pcg),
                     mapped_pcg.machine_mapping);
}

std::ostream &operator<<(std::ostream &s, MappedParallelComputationGraph const &mapped_pcg) {
  return (s << fmt::to_string(mapped_pcg));
}

} // namespace FlexFlow
