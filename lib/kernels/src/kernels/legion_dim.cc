#include "kernels/legion_dim.h"
#include "utils/containers/range.h"
#include "utils/containers/transform.h"
#include "utils/containers/set_of.h"

namespace FlexFlow {

std::set<legion_dim_t> legion_dim_range(int end) {
  return set_of(transform(range(end), [](int i) { return ff_dim_t{i}; }));
}

legion_dim_t add_to_legion_dim(legion_dim_t legion_dim, int value) {
  return legion_dim_t(legion_dim.value + value);
}

legion_dim_t legion_dim_from_ff_dim(ff_dim_t ff_dim, int num_dimensions) {
  return legion_dim_t(num_dimensions - ff_dim.value - 1);
}

} // namespace FlexFlow
