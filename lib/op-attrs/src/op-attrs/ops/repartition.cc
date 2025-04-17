#include "op-attrs/ops/repartition.h"

namespace FlexFlow {

RecordFormatter as_dot(RepartitionAttrs const &attrs) {
  RecordFormatter r;

  auto kv = [](std::string const &label, auto const &val) {
    RecordFormatter rr;
    rr << label << fmt::to_string(val);
    return rr;
  };

  r << kv("dim", attrs.repartition_dim)
    << kv("degree", attrs.repartition_degree);

  return r;
}

tl::expected<ParallelTensorShape, std::string>
    get_output_shape(RepartitionAttrs const &attrs,
                     ParallelTensorShape const &input_shape) {
  ParallelTensorShape output_shape = input_shape;
  output_shape.dims.shard_dims
      .at(relative_ff_dim_t_from_ff_dim_t(attrs.repartition_dim))
      .degree *= attrs.repartition_degree;
  return output_shape;
}

} // namespace FlexFlow
