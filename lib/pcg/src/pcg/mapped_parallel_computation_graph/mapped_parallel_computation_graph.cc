#include "pcg/mapped_parallel_computation_graph/mapped_parallel_computation_graph.h"

namespace FlexFlow {

std::string format_as(MappedParallelComputationGraph const &mapped_pcg) {
  return fmt::format("<MappedParallelComputationGraph\npcg={}\nmapped_tasks={}>",
                     as_dot(mapped_pcg.pcg),
                     mapped_pcg.mapped_tasks);
}

std::ostream &operator<<(std::ostream &s, MappedParallelComputationGraph const &mapped_pcg) {
  return (s << fmt::to_string(mapped_pcg));
}


} // namespace FlexFlow
