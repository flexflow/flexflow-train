#include "utils/containers/compute_mean.h"
#include "utils/containers/sum.h"
#include <libassert/assert.hpp>

namespace FlexFlow {

float compute_mean(std::vector<float> const &vs) {
  ASSERT(!vs.empty());

  return sum(vs) / vs.size();
}

} // namespace FlexFlow
