#include "op-attrs/relative_ff_dim_t.h"
#include "rapidcheck.h"

namespace FlexFlow {
ff_dim_t ff_dim_t_from_relative_ff_dim_t(relative_ff_dim_t ff_dim,
                                         int input_dim) {
  int raw = ff_dim.value;
  if (raw < 0) {
    raw = input_dim + raw;
  }
  return ff_dim_t{nonnegative_int{raw}};
}
} // namespace FlexFlow

namespace rc {
Gen<::FlexFlow::relative_ff_dim_t>
    Arbitrary<::FlexFlow::relative_ff_dim_t>::arbitrary() {
  return gen::construct<::FlexFlow::relative_ff_dim_t>(
      gen::inRange<int>(-MAX_TENSOR_DIM, MAX_TENSOR_DIM));
}
} // namespace rc
