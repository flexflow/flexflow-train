#include "compiler/graph_optimize_result.h"

namespace FlexFlow {

std::string format_as(GraphOptimizeResult const &r) {
  return fmt::format("<GraphOptimizeResult\npcg={}\nmachine_mapping={}>",
                     as_dot(r.pcg),
                     r.machine_mapping);
}

std::ostream &operator<<(std::ostream &s, GraphOptimizeResult const &r) {
  return (s << fmt::to_string(r));
}

} // namespace FlexFlow
