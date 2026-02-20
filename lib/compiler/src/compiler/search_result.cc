#include "compiler/search_result.h"

namespace FlexFlow {

MappedParallelComputationGraph
    get_mapped_pcg_from_search_result(SearchResult const &search_result) {
  return mapped_pcg_from_pcg_and_mapping(search_result.pcg,
                                         search_result.machine_mapping);
}

std::string format_as(SearchResult const &r) {
  return fmt::format("<SearchResult\npcg={}\nmachine_mapping={}>",
                     as_dot(r.pcg),
                     r.machine_mapping);
}

std::ostream &operator<<(std::ostream &s, SearchResult const &r) {
  return (s << fmt::to_string(r));
}

} // namespace FlexFlow
