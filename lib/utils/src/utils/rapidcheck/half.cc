#include "utils/rapidcheck/half.h"

namespace rc {

Gen<::half> Arbitrary<::half>::arbitrary() {
  return gen::construct<::half>(gen::arbitrary<float>());
}

} // namespace rc
