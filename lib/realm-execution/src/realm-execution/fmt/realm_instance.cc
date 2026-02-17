#include "realm-execution/fmt/realm_instance.h"

namespace FlexFlow {

std::ostream &operator<<(std::ostream &s,
                         ::FlexFlow::Realm::RegionInstance const &m) {
  return s << fmt::to_string(m);
}

} // namespace FlexFlow
