#include "realm-execution/fmt/realm_processor.h"

namespace FlexFlow {

std::ostream &operator<<(std::ostream &s,
                         ::FlexFlow::Realm::Processor const &m) {
  return s << fmt::to_string(m);
}

} // namespace FlexFlow
