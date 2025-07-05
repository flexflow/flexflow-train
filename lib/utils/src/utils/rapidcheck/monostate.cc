#include "utils/rapidcheck/monostate.h"

namespace rc {

Gen<std::monostate> Arbitrary<std::monostate>::arbitrary() {
  return gen::construct<std::monostate>();
}

} // namespace rc
