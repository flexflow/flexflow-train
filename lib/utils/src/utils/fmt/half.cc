#include "utils/fmt/half.h"

namespace FlexFlow {

std::ostream &operator<<(std::ostream &s, ::half h) {
  return (s << static_cast<float>(h));
}

} // namespace FlexFlow
