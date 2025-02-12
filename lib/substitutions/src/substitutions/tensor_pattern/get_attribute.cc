#include "substitutions/tensor_pattern/get_attribute.h"
#include "op-attrs/parallel_tensor_dims.h"
#include "utils/containers/transform.h"
#include "utils/containers/vector_of.h"
#include "utils/integer_conversions.h"

namespace FlexFlow {

TensorAttributeValue get_attribute(ParallelTensorAttrs const &attrs,
                                   TensorAttributeKey key) {
  switch (key) {
    case TensorAttributeKey::DIM_SIZES: {
      std::vector<nonnegative_int> sizes = transform(
          vector_of(ff_ordered_shard_dims(attrs.shape.dims)),
          [](ShardParallelDim const &d) { return nonnegative_int{d.size}; });
      return TensorAttributeValue{sizes};
    }
    case TensorAttributeKey::DIM_DEGREES: {
      std::vector<nonnegative_int> degrees = transform(
          vector_of(ff_ordered_shard_dims(attrs.shape.dims)),
          [](ShardParallelDim const &d) { return nonnegative_int{d.degree}; });
      return TensorAttributeValue{degrees};
    }
    default:
      throw std::runtime_error(
          fmt::format("Unknown TensorAttributeKey {}", static_cast<int>(key)));
  }
}

} // namespace FlexFlow
