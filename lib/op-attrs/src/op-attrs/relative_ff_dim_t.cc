#include "op-attrs/relative_ff_dim_t.h"
#include "rapidcheck.h"

namespace rc {
Gen<FlexFlow::relative_ff_dim_t>
    Arbitrary<FlexFlow::relative_ff_dim_t>::arbitrary() {
  return gen::construct<FlexFlow::relative_ff_dim_t>(
      gen::inRange<int>(-MAX_TENSOR_DIM, MAX_TENSOR_DIM));
}
} // namespace rc
