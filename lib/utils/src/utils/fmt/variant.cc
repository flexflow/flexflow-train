#include "utils/fmt/variant.h"

namespace FlexFlow {

template
  std::ostream &operator<<(std::ostream &, std::variant<int> const &);

template
  std::ostream &operator<<(std::ostream &, std::variant<std::string, int> const &);

} // namespace FlexFlow
