#include "local-execution/legion_tensor_shape.h"
#include "kernels/legion_dim.h"
#include "op-attrs/tensor_shape.h"

namespace FlexFlow {

legion_dim_t legion_dim_from_ff_dim(ff_dim_t ff_dim, size_t num_dims) {
  return legion_dim_t(num_dims - ff_dim.value.get_value() - 1);
}

legion_dim_t legion_dim_from_ff_dim(ff_dim_t ff_dim, TensorShape const &shape) {
  return legion_dim_from_ff_dim(ff_dim, num_dims(shape));
}

} // namespace FlexFlow
