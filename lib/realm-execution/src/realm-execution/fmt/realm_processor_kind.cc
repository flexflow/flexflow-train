#include "realm-execution/fmt/realm_processor_kind.h"

namespace FlexFlow {

std::ostream &operator<<(std::ostream &s,
                         ::FlexFlow::Realm::Processor::Kind const &k) {
  return (s << fmt::to_string(k));
}

} // namespace FlexFlow
