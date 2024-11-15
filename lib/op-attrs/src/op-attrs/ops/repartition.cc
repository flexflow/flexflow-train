#include "op-attrs/ops/repartition.h"

namespace FlexFlow {

tl::expected<ParallelTensorShape, std::string>
    get_output_shape(RepartitionAttrs const &attrs,
                     ParallelTensorShape const &input_shape) {
  ParallelTensorShape output_shape = input_shape;
  output_shape.dims.shard_dims
      .at(relative_ff_dim_t{attrs.repartition_dim.value.get_value()})
      .degree *= attrs.repartition_degree;
  return output_shape;
}

} // namespace FlexFlow
