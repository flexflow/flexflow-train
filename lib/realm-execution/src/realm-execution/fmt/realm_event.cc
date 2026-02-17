#include "realm-execution/fmt/realm_event.h"

namespace FlexFlow {

std::ostream &operator<<(std::ostream &s,
                         ::FlexFlow::Realm::Event const &m) {
  return s << fmt::to_string(m);
}

} // namespace FlexFlow
