#include "compiler/search_result.h"

namespace FlexFlow {

std::string format_as(SearchResult const &r) {
  return fmt::format("<SearchResult\npcg={}\nmachine_mapping={}>",
                     as_dot(r.pcg),
                     r.machine_mapping);
}

std::ostream &operator<<(std::ostream &s, SearchResult const &r) {
  return (s << fmt::to_string(r));
}

} // namespace FlexFlow
