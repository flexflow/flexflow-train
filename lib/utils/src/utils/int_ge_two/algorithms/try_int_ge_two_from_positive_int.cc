#include "utils/int_ge_two/algorithms/try_int_ge_two_from_positive_int.h"

namespace FlexFlow {

std::optional<int_ge_two> try_int_ge_two_from_positive_int(positive_int p) {
  if (p == 1) {
    return std::nullopt;
  } else {
    return int_ge_two{p};
  }
}

} // namespace FlexFlow
