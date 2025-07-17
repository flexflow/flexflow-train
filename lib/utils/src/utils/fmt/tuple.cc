#include "utils/fmt/tuple.h"

namespace FlexFlow {

template std::ostream &operator<<(std::ostream &s,
                                  std::tuple<int, char, std::string> const &);

} // namespace FlexFlow
