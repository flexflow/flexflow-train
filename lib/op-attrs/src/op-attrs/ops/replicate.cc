#include "op-attrs/ops/replicate.h"

namespace FlexFlow {

RecordFormatter as_dot(ReplicateAttrs const &attrs) {
  RecordFormatter r;

  auto kv = [](std::string const &label, auto const &val) {
    RecordFormatter rr;
    rr << label << fmt::to_string(val);
    return rr;
  };

  r << kv("degree", attrs.replicate_degree);

  return r;
}

ParallelTensorShape get_output_shape(ReplicateAttrs const &attrs,
                                     ParallelTensorShape const &input_shape) {
  ParallelTensorShape output_shape = input_shape;
  output_shape.dims.replica_dims.discard_copy_degree.value *=
      attrs.replicate_degree;
  return output_shape;
}

} // namespace FlexFlow
