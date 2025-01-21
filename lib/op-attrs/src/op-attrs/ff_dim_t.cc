#include "op-attrs/ff_dim_t.h"

namespace FlexFlow {
relative_ff_dim_t relative_ff_dim_t_from_ff_dim_t(ff_dim_t ff_dim) {
  return relative_ff_dim_t{ff_dim.value.get_value()};
}
} // namespace FlexFlow

namespace rc {
Gen<::FlexFlow::ff_dim_t> Arbitrary<::FlexFlow::ff_dim_t>::arbitrary() {
  return gen::construct<::FlexFlow::ff_dim_t>(
      gen::map(gen::inRange<int>(0, MAX_TENSOR_DIM),
               [](int value) { return FlexFlow::nonnegative_int{value}; }));
}
} // namespace rc
