#include "op-attrs/tensor_shape.h"
#include "op-attrs/datatype.h"
#include "op-attrs/tensor_dims.h"
#include "utils/containers/get_only.h"
#include "utils/containers/product.h"
#include "utils/containers/transform.h"
#include "utils/nonnegative_int/num_elements.h"

namespace FlexFlow {

num_bytes_t get_size_in_bytes(TensorShape const &s) {
  return num_bytes_t{
    (get_num_elements(s.dims) * size_of_datatype(s.data_type)).nonnegative_int_from_positive_int()
  }; 
}

TensorShape tensor_shape_drop_dims(
    TensorShape const &coord,
    std::function<bool(ff_dim_t)> const &should_drop_dim) {
  NOT_IMPLEMENTED();
}

TensorShape slice_tensor_shape(TensorShape const &shape,
                               relative_ff_dim_t const &start,
                               std::optional<relative_ff_dim_t> const &stop) {
  return TensorShape{
      slice_tensor_dims(shape.dims, start, stop),
      shape.data_type,
  };
}

} // namespace FlexFlow
