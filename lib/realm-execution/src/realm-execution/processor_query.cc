#include "realm-execution/processor_query.h"

namespace FlexFlow {

std::set<Realm::Processor>
    processor_set_from_query(Realm::Machine::ProcessorQuery const &pq) {
  std::set<Realm::Processor> result;
  for (Realm::Processor p : pq) {
    result.insert(p);
  }
  return result;
}

} // namespace FlexFlow
