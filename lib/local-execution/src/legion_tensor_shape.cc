#include "local-execution/legion_tensor_shape.h"
#include "op-attrs/tensor_shape.h"

namespace FlexFlow {

legion_dim_t legion_dim_from_ff_dim_t(ff_dim_t ff_dim_t, size_t num_dims) {
  return legion_dim_t(num_dims - ff_dim_t.value - 1);
}

legion_dim_t legion_dim_from_ff_dim_t(ff_dim_t ff_dim_t,
                                      TensorShape const &shape) {
  return legion_dim_t(num_dims(shape) - ff_dim_t.value - 1);
}

} // namespace FlexFlow
