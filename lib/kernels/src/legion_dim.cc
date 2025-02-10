#include "kernels/legion_dim.h"

namespace FlexFlow {

legion_dim_t add_to_legion_dim(legion_dim_t legion_dim, int value) {
  return legion_dim_t{
      nonnegative_int{legion_dim.value.unwrap_nonnegative() + value}};
}

legion_dim_t legion_dim_from_ff_dim(ff_dim_t ff_dim,
                                    nonnegative_int num_dimensions) {
  return legion_dim_t{num_dimensions - ff_dim.value - 1_n};
  ;
}

} // namespace FlexFlow
