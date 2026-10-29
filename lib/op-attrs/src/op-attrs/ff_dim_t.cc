#include "op-attrs/ff_dim_t.h"
#include "utils/containers/transform.h"
#include "utils/nonnegative_int/nonnegative_range.h"
#include "utils/nonnegative_int/nonnegative_range_inclusive.h"
#include <libassert/assert.hpp>

namespace FlexFlow {

relative_ff_dim_t relative_ff_dim_t_from_ff_dim_t(ff_dim_t ff_dim) {

  return relative_ff_dim_t{ff_dim.value.unwrap_nonnegative()};
}

ff_dim_t add_to_ff_dim(ff_dim_t ff_dim, int value) {
  return ff_dim_t{nonnegative_int{ff_dim.value.unwrap_nonnegative() + value}};
}

std::vector<ff_dim_t> ff_dim_range(nonnegative_int num_elements) {
  return transform(nonnegative_range(num_elements),
                   [](nonnegative_int idx) { return ff_dim_t{idx}; });
}

std::vector<ff_dim_t> ff_dim_range2_inclusive(ff_dim_t start, ff_dim_t stop) {
  return transform(nonnegative_range_inclusive(start.value, stop.value),
                   [&](nonnegative_int i) -> ff_dim_t { return ff_dim_t{i}; });
}

std::vector<ff_dim_t> ff_dim_range2_exclusive(ff_dim_t start, ff_dim_t stop) {
  return transform(nonnegative_range(start.value, stop.value),
                   [&](nonnegative_int i) -> ff_dim_t { return ff_dim_t{i}; });
}

} // namespace FlexFlow

namespace rc {

Gen<::FlexFlow::ff_dim_t> Arbitrary<::FlexFlow::ff_dim_t>::arbitrary() {
  return gen::construct<::FlexFlow::ff_dim_t>(
      gen::map(gen::inRange<int>(0, MAX_TENSOR_DIM),
               [](int value) { return FlexFlow::nonnegative_int{value}; }));
}

} // namespace rc
