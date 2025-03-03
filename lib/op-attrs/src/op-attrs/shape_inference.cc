#include "op-attrs/shape_inference.h"
#include "op-attrs/ops/attention.h"
#include "op-attrs/ops/batch_matmul.h"
#include "op-attrs/ops/batch_norm.h"
#include "op-attrs/ops/cast.h"
#include "op-attrs/ops/combine.h"
#include "op-attrs/ops/concat.h"
#include "op-attrs/ops/conv_2d.h"
#include "op-attrs/ops/dropout.h"
#include "op-attrs/ops/element_binary.h"
#include "op-attrs/ops/element_unary.h"
#include "op-attrs/ops/embedding.h"
#include "op-attrs/ops/flat.h"
#include "op-attrs/ops/gather.h"
#include "op-attrs/ops/input.h"
#include "op-attrs/ops/layer_norm.h"
#include "op-attrs/ops/linear.h"
#include "op-attrs/ops/pool_2d.h"
#include "op-attrs/ops/reduction.h"
#include "op-attrs/ops/repartition.h"
#include "op-attrs/ops/replicate.h"
#include "op-attrs/ops/softmax.h"
#include "op-attrs/ops/weight.h"
#include "utils/containers/get_only.h"
#include "utils/overload.h"

namespace FlexFlow {

template <typename T>
static std::pair<T, T> require_2(std::vector<T> const &v) {
  assert(v.size() == 2);

  return {v.at(0), v.at(1)};
}

template <typename T>
static std::tuple<T, T, T> require_3(std::vector<T> const &v) {
  assert(v.size() == 3);

  return {v.at(0), v.at(1), v.at(2)};
}

std::vector<TensorShape>
    get_output_shapes(ComputationGraphOpAttrs const &op_attrs,
                      std::vector<TensorShape> const &input_shapes) {
  return op_attrs.visit<std::vector<TensorShape>>(overload{
      [&](BatchMatmulAttrs const &attrs) -> std::vector<TensorShape> {
        auto [i1, i2] = require_2(input_shapes);

        return {throw_if_unexpected(get_output_shape(attrs, i1, i2))};
      },
      [&](BatchNormAttrs const &attrs) -> std::vector<TensorShape> {
        return {throw_if_unexpected(
            get_output_shape(attrs, get_only(input_shapes)))};
      },
      [&](CastAttrs const &attrs) -> std::vector<TensorShape> {
        return {throw_if_unexpected(
            get_output_shape(attrs, get_only(input_shapes)))};
      },
      [&](ConcatAttrs const &attrs) -> std::vector<TensorShape> {
        return {throw_if_unexpected(get_output_shape(attrs, input_shapes))};
      },
      [&](Conv2DAttrs const &attrs) -> std::vector<TensorShape> {
        return {get_output_shape(attrs, get_only(input_shapes))};
      },
      [&](DropoutAttrs const &attrs) -> std::vector<TensorShape> {
        return {get_output_shape(attrs, get_only(input_shapes))};
      },
      [&](ElementBinaryAttrs const &attrs) -> std::vector<TensorShape> {
        auto [i1, i2] = require_2(input_shapes);

        return {throw_if_unexpected(get_output_shape(attrs, i1, i2))};
      },
      [&](ElementUnaryAttrs const &attrs) -> std::vector<TensorShape> {
        return {throw_if_unexpected(
            get_output_shape(attrs, get_only(input_shapes)))};
      },
      [&](EmbeddingAttrs const &attrs) -> std::vector<TensorShape> {
        return {throw_if_unexpected(
            get_output_shape(attrs, get_only(input_shapes)))};
      },
      [&](FlatAttrs const &attrs) -> std::vector<TensorShape> {
        return {get_output_shape(attrs, get_only(input_shapes))};
      },
      [&](GatherAttrs const &attrs) -> std::vector<TensorShape> {
        return {
            get_output_shape(attrs, input_shapes.at(0), input_shapes.at(1))};
      },
      [&](InputAttrs const &attrs) -> std::vector<TensorShape> {
        return {get_output_shape(attrs)};
      },
      [&](LayerNormAttrs const &attrs) -> std::vector<TensorShape> {
        return {throw_if_unexpected(
            get_output_shape(attrs, get_only(input_shapes)))};
      },
      [&](LinearAttrs const &attrs) -> std::vector<TensorShape> {
        return {throw_if_unexpected(
            get_output_shape(attrs, get_only(input_shapes)))};
      },
      [&](MultiHeadAttentionAttrs const &attrs) -> std::vector<TensorShape> {
        auto [i1, i2, i3] = require_3(input_shapes);

        return {throw_if_unexpected(get_output_shape(attrs, i1, i2, i3))};
      },
      [&](Pool2DAttrs const &attrs) -> std::vector<TensorShape> {
        return {throw_if_unexpected(
            get_output_shape(attrs, get_only(input_shapes)))};
      },
      [&](SoftmaxAttrs const &attrs) -> std::vector<TensorShape> {
        return {throw_if_unexpected(
            get_output_shape(attrs, get_only(input_shapes)))};
      },
      [&](WeightAttrs const &attrs) -> std::vector<TensorShape> {
        return {get_output_shape(attrs)};
      },
      [&](auto const &attrs) -> std::vector<TensorShape> {
        NOT_IMPLEMENTED();
      }});
}

std::vector<TensorShape>
    get_weight_shapes(ComputationGraphOpAttrs const &op_attrs,
                      std::vector<TensorShape> const &input_shapes) {
  return op_attrs.visit<std::vector<TensorShape>>(overload{
      [&](BatchMatmulAttrs const &attrs) -> std::vector<TensorShape> {
        return {};
      },
      [&](BatchNormAttrs const &attrs) -> std::vector<TensorShape> {
        return throw_if_unexpected(
            get_weight_shapes(attrs, get_only(input_shapes)));
      },
      [&](CastAttrs const &attrs) -> std::vector<TensorShape> { return {}; },
      [&](ConcatAttrs const &attrs) -> std::vector<TensorShape> { return {}; },
      [&](Conv2DAttrs const &attrs) -> std::vector<TensorShape> {
        return get_weight_shapes(attrs, get_only(input_shapes));
      },
      [&](DropoutAttrs const &attrs) -> std::vector<TensorShape> { return {}; },
      [&](ElementBinaryAttrs const &attrs) -> std::vector<TensorShape> {
        return {};
      },
      [&](ElementUnaryAttrs const &attrs) -> std::vector<TensorShape> {
        return {};
      },
      [&](EmbeddingAttrs const &attrs) -> std::vector<TensorShape> {
        return {throw_if_unexpected(
            get_weights_shape(attrs, get_only(input_shapes)))};
      },
      [&](FlatAttrs const &attrs) -> std::vector<TensorShape> { return {}; },
      [&](GatherAttrs const &attrs) -> std::vector<TensorShape> { return {}; },
      [&](InputAttrs const &attrs) -> std::vector<TensorShape> { return {}; },
      [&](LayerNormAttrs const &attrs) -> std::vector<TensorShape> {
        return throw_if_unexpected(
            get_weight_shapes(attrs, get_only(input_shapes)));
      },
      [&](LinearAttrs const &attrs) -> std::vector<TensorShape> {
        return throw_if_unexpected(
            get_weight_shapes(attrs, get_only(input_shapes)));
      },
      [&](MultiHeadAttentionAttrs const &attrs) -> std::vector<TensorShape> {
        auto [i1, i2, i3] = require_3(input_shapes);

        return throw_if_unexpected(get_weight_shapes(attrs, i1, i2, i3));
      },
      [&](Pool2DAttrs const &attrs) -> std::vector<TensorShape> { return {}; },
      [&](SoftmaxAttrs const &attrs) -> std::vector<TensorShape> { return {}; },
      [&](WeightAttrs const &attrs) -> std::vector<TensorShape> { return {}; },
      [&](auto const &attrs) -> std::vector<TensorShape> {
        NOT_IMPLEMENTED();
      }});
}

std::vector<ParallelTensorShape>
    get_output_shapes(PCGOperatorAttrs const &pcg_op_attrs,
                      std::vector<ParallelTensorShape> const &input_shapes) {
  return pcg_op_attrs.visit<std::vector<ParallelTensorShape>>(overload{
      [&](BatchMatmulAttrs const &attrs) -> std::vector<ParallelTensorShape> {
        auto [i1, i2] = require_2(input_shapes);

        return {throw_if_unexpected(get_output_shape(attrs, i1, i2))};
      },
      [&](BatchNormAttrs const &attrs) -> std::vector<ParallelTensorShape> {
        return {throw_if_unexpected(
            get_output_shape(attrs, get_only(input_shapes)))};
      },
      [&](CastAttrs const &attrs) -> std::vector<ParallelTensorShape> {
        return {throw_if_unexpected(
            get_output_shape(attrs, get_only(input_shapes)))};
      },
      [&](CombineAttrs const &attrs) -> std::vector<ParallelTensorShape> {
        return {throw_if_unexpected(
            get_output_shape(attrs, get_only(input_shapes)))};
      },
      [&](ConcatAttrs const &attrs) -> std::vector<ParallelTensorShape> {
        return {throw_if_unexpected(get_output_shape(attrs, input_shapes))};
      },
      [&](Conv2DAttrs const &attrs) -> std::vector<ParallelTensorShape> {
        return {get_output_shape(attrs, get_only(input_shapes))};
      },
      [&](DropoutAttrs const &attrs) -> std::vector<ParallelTensorShape> {
        return {throw_if_unexpected(
            get_output_shape(attrs, get_only(input_shapes)))};
      },
      [&](ElementBinaryAttrs const &attrs) -> std::vector<ParallelTensorShape> {
        auto [i1, i2] = require_2(input_shapes);

        return {throw_if_unexpected(get_output_shape(attrs, i1, i2))};
      },
      [&](ElementUnaryAttrs const &attrs) -> std::vector<ParallelTensorShape> {
        return {throw_if_unexpected(
            get_output_shape(attrs, get_only(input_shapes)))};
      },
      [&](EmbeddingAttrs const &attrs) -> std::vector<ParallelTensorShape> {
        return {throw_if_unexpected(
            get_output_shape(attrs, get_only(input_shapes)))};
      },
      [&](FlatAttrs const &attrs) -> std::vector<ParallelTensorShape> {
        return {throw_if_unexpected(
            get_output_shape(attrs, get_only(input_shapes)))};
      },
      [&](GatherAttrs const &attrs) -> std::vector<ParallelTensorShape> {
        return {
            get_output_shape(attrs, input_shapes.at(0), input_shapes.at(1))};
      },
      [&](InputAttrs const &attrs) -> std::vector<ParallelTensorShape> {
        return {get_output_parallel_tensor_shape(attrs)};
      },
      [&](LayerNormAttrs const &attrs) -> std::vector<ParallelTensorShape> {
        return {throw_if_unexpected(
            get_output_shape(attrs, get_only(input_shapes)))};
      },
      [&](LinearAttrs const &attrs) -> std::vector<ParallelTensorShape> {
        return {throw_if_unexpected(
            get_output_shape(attrs, get_only(input_shapes)))};
      },
      [&](MultiHeadAttentionAttrs const &attrs)
          -> std::vector<ParallelTensorShape> {
        auto [i1, i2, i3] = require_3(input_shapes);

        return {throw_if_unexpected(get_output_shape(attrs, i1, i2, i3))};
      },
      [&](Pool2DAttrs const &attrs) -> std::vector<ParallelTensorShape> {
        return {throw_if_unexpected(
            get_output_shape(attrs, get_only(input_shapes)))};
      },
      [&](ReductionAttrs const &attrs) -> std::vector<ParallelTensorShape> {
        return {throw_if_unexpected(
            get_output_shape(attrs, get_only(input_shapes)))};
      },
      [&](RepartitionAttrs const &attrs) -> std::vector<ParallelTensorShape> {
        return {throw_if_unexpected(
            get_output_shape(attrs, get_only(input_shapes)))};
      },
      [&](ReplicateAttrs const &attrs) -> std::vector<ParallelTensorShape> {
        return {get_output_shape(attrs, get_only(input_shapes))};
      },
      [&](SoftmaxAttrs const &attrs) -> std::vector<ParallelTensorShape> {
        return {throw_if_unexpected(
            get_output_shape(attrs, get_only(input_shapes)))};
      },
      [&](WeightAttrs const &attrs) -> std::vector<ParallelTensorShape> {
        return {get_output_parallel_tensor_shape(attrs)};
      },
      [&](auto const &attrs) -> std::vector<ParallelTensorShape> {
        NOT_IMPLEMENTED();
      }});
}

std::vector<ParallelTensorShape>
    get_weight_shapes(PCGOperatorAttrs const &pcg_op_attrs,
                      std::vector<ParallelTensorShape> const &input_shapes) {
  return pcg_op_attrs.visit<std::vector<ParallelTensorShape>>(overload{
      [&](BatchMatmulAttrs const &attrs) -> std::vector<ParallelTensorShape> {
        return {};
      },
      [&](BatchNormAttrs const &attrs) -> std::vector<ParallelTensorShape> {
        return throw_if_unexpected(
            get_weight_shapes(attrs, get_only(input_shapes)));
      },
      [&](CastAttrs const &attrs) -> std::vector<ParallelTensorShape> {
        return {};
      },
      [&](CombineAttrs const &attrs) -> std::vector<ParallelTensorShape> {
        return {};
      },
      [&](ConcatAttrs const &attrs) -> std::vector<ParallelTensorShape> {
        return {};
      },
      [&](Conv2DAttrs const &attrs) -> std::vector<ParallelTensorShape> {
        return get_weight_shapes(attrs, get_only(input_shapes));
      },
      [&](DropoutAttrs const &attrs) -> std::vector<ParallelTensorShape> {
        return {};
      },
      [&](ElementBinaryAttrs const &attrs) -> std::vector<ParallelTensorShape> {
        return {};
      },
      [&](ElementUnaryAttrs const &attrs) -> std::vector<ParallelTensorShape> {
        return {};
      },
      [&](EmbeddingAttrs const &attrs) -> std::vector<ParallelTensorShape> {
        return {
            throw_if_unexpected(
                get_weights_shape(attrs, get_only(input_shapes))),
        };
      },
      [&](FlatAttrs const &attrs) -> std::vector<ParallelTensorShape> {
        return {};
      },
      [&](GatherAttrs const &attrs) -> std::vector<ParallelTensorShape> {
        return {};
      },
      [&](InputAttrs const &attrs) -> std::vector<ParallelTensorShape> {
        return {};
      },
      [&](LayerNormAttrs const &attrs) -> std::vector<ParallelTensorShape> {
        return throw_if_unexpected(
            get_weight_shapes(attrs, get_only(input_shapes)));
      },
      [&](LinearAttrs const &attrs) -> std::vector<ParallelTensorShape> {
        return throw_if_unexpected(
            get_weight_shapes(attrs, get_only(input_shapes)));
      },
      [&](MultiHeadAttentionAttrs const &attrs)
          -> std::vector<ParallelTensorShape> {
        auto [i1, i2, i3] = require_3(input_shapes);

        return throw_if_unexpected(get_weight_shapes(attrs, i1, i2, i3));
      },
      [&](Pool2DAttrs const &attrs) -> std::vector<ParallelTensorShape> {
        return {};
      },
      [&](RepartitionAttrs const &attrs) -> std::vector<ParallelTensorShape> {
        return {};
      },
      [&](ReplicateAttrs const &attrs) -> std::vector<ParallelTensorShape> {
        return {};
      },
      [&](ReductionAttrs const &attrs) -> std::vector<ParallelTensorShape> {
        return {};
      },
      [&](SoftmaxAttrs const &attrs) -> std::vector<ParallelTensorShape> {
        return {};
      },
      [&](WeightAttrs const &attrs) -> std::vector<ParallelTensorShape> {
        return {};
      },
      [&](auto const &attrs) -> std::vector<ParallelTensorShape> {
        NOT_IMPLEMENTED();
      }});
}

} // namespace FlexFlow
