#include "op-attrs/ops/cast.h"
#include "op-attrs/datatype.h"

namespace FlexFlow {

tl::expected<TensorShape, std::string>
    get_output_shape(CastAttrs const &attrs, TensorShape const &input) {

  TensorShape output = input;
  output.data_type = attrs.dtype;
  return output;
}

tl::expected<ParallelTensorShape, std::string>
    get_output_shape(CastAttrs const &attrs, ParallelTensorShape const &input) {

  ParallelTensorShape output = input;
  output.data_type = attrs.dtype;
  return output;
}

} // namespace FlexFlow
