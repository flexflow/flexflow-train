#include "op-attrs/tensor_shape.h"
#include "op-attrs/datatype.h"
#include "op-attrs/tensor_dims.h"
#include "utils/containers/get_only.h"
#include "utils/containers/product.h"
#include "utils/containers/transform.h"
#include "utils/nonnegative_int/num_elements.h"

namespace FlexFlow {

nonnegative_int num_dims(TensorShape const &s) {
  return num_elements(s.dims.ff_ordered);
}

nonnegative_int dim_at_idx(TensorShape const &s, relative_ff_dim_t idx) {
  return dim_at_idx(s.dims, idx);
}

nonnegative_int &dim_at_idx(TensorShape &s, relative_ff_dim_t idx) {
  return dim_at_idx(s.dims, idx);
}

nonnegative_int get_num_elements(TensorShape const &s) {
  return get_num_elements(s.dims);
}

nonnegative_int get_size_in_bytes(TensorShape const &s) {
  return get_num_elements(s) * size_of_datatype(s.data_type);
}

TensorShape slice_tensor_shape(TensorShape const &shape,
                               std::optional<relative_ff_dim_t> const &start,
                               std::optional<relative_ff_dim_t> const &stop) {
  return TensorShape{
      slice_tensor_dims(shape.dims, start, stop),
      shape.data_type,
  };
}

} // namespace FlexFlow
