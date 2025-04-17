#include "op-attrs/ops/cast.h"
#include "op-attrs/datatype.h"

namespace FlexFlow {

RecordFormatter as_dot(CastAttrs const &attrs) {
  RecordFormatter r;

  auto kv = [](std::string const &label, auto const &val) {
    RecordFormatter rr;
    rr << label << fmt::to_string(val);
    return rr;
  };

  r << kv("to", attrs.dtype);

  return r;
}

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
