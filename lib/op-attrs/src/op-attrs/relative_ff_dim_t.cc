#include "op-attrs/relative_ff_dim_t.h"
#include <libassert/assert.hpp>
#include <rapidcheck.h>

namespace FlexFlow {

ff_dim_t ff_dim_t_from_relative_ff_dim_t(relative_ff_dim_t ff_dim,
                                         num_tensor_dims_t num_dims) {
  int raw = ff_dim.value;
  if (raw < 0) {
    raw = num_dims.int_from_num_tensor_dims() + raw;
  }

  ff_dim_t result = ff_dim_t{nonnegative_int{raw}};
  ASSERT(ff_dim_is_valid_for_num_tensor_dims(result, num_dims));

  return result;
}

} // namespace FlexFlow

namespace rc {

Gen<::FlexFlow::relative_ff_dim_t>
    Arbitrary<::FlexFlow::relative_ff_dim_t>::arbitrary() {
  return gen::construct<::FlexFlow::relative_ff_dim_t>(
      gen::inRange<int>(-MAX_TENSOR_DIM, MAX_TENSOR_DIM));
}

} // namespace rc
