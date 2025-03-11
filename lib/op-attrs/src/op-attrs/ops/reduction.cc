#include "op-attrs/ops/reduction.h"
#include "op-attrs/parallel_tensor_shape.h"

namespace FlexFlow {

RecordFormatter as_dot(ReductionAttrs const &attrs) {
  RecordFormatter r;

  auto kv = [](std::string const &label, auto const &val) {
    RecordFormatter rr;
    rr << label << fmt::to_string(val);
    return rr;
  };

  r << kv("degree", attrs.reduction_degree);

  return r;
}

tl::expected<ParallelTensorShape, std::string>
    get_output_shape(ReductionAttrs const &attrs,
                     ParallelTensorShape const &input_shape) {
  if (get_sum_degree(input_shape) % attrs.reduction_degree != 0) {
    return tl::unexpected(
        fmt::format("Reduction received tensor with sum degree {}, which is "
                    "not divisible by reduction degree {}",
                    get_sum_degree(input_shape),
                    attrs.reduction_degree));
  }

  ParallelTensorShape output_shape = input_shape;
  output_shape.dims.replica_dims.sum_degree.value /= attrs.reduction_degree;
  return output_shape;
}

} // namespace FlexFlow
