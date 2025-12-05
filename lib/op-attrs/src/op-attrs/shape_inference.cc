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
#include "op-attrs/ops/transpose.h"
#include "op-attrs/ops/weight.h"
#include "utils/containers/get_only.h"
#include "utils/overload.h"
#include "utils/containers/require_only_key.h"
#include "utils/containers/require_two_keys.h"

namespace FlexFlow {

template <typename T>
static std::tuple<T, T, T> require_3(std::unordered_map<TensorSlotName, T> const &v, TensorSlotName k1, TensorSlotName k2, TensorSlotName k3) {
  ASSERT(v.size() == 3);

  return {v.at(k1), v.at(k2), v.at(k3)};
}

std::unordered_map<TensorSlotName, SingularOrVariadic<TensorShape>>
    get_output_shapes(ComputationGraphOpAttrs const &op_attrs,
                      std::unordered_map<TensorSlotName, SingularOrVariadic<TensorShape>> const &input_shapes) {
  return op_attrs.visit<std::unordered_map<TensorSlotName, SingularOrVariadic<TensorShape>>>(overload{
      [&](BatchMatmulAttrs const &attrs) -> std::unordered_map<TensorSlotName, SingularOrVariadic<TensorShape>> {
        auto [lhs, rhs] = require_two_keys(input_shapes, TensorSlotName::LHS_INPUT, TensorSlotName::RHS_INPUT);

        return {
          {
            TensorSlotName::OUTPUT,
            SingularOrVariadic<TensorShape>{
              throw_if_unexpected(get_output_shape(attrs, lhs.require_singular(), rhs.require_singular())),
            },
          },
        };
      },
      [&](BatchNormAttrs const &attrs) -> std::unordered_map<TensorSlotName, SingularOrVariadic<TensorShape>> {
        TensorShape input = require_only_key(input_shapes, TensorSlotName::INPUT).require_singular();

        return {
          {
            TensorSlotName::OUTPUT,
            SingularOrVariadic<TensorShape>{
              throw_if_unexpected(get_output_shape(attrs, input)),
            },
          },
        };
      },
      [&](CastAttrs const &attrs) -> std::unordered_map<TensorSlotName, SingularOrVariadic<TensorShape>> {
        TensorShape input = require_only_key(input_shapes, TensorSlotName::INPUT).require_singular();

        return {
          {
            TensorSlotName::OUTPUT,
            SingularOrVariadic<TensorShape>{
              throw_if_unexpected(get_output_shape(attrs, input)),
            },
          },
        };
      },
      [&](ConcatAttrs const &attrs) -> std::unordered_map<TensorSlotName, SingularOrVariadic<TensorShape>> {
        std::vector<TensorShape> inputs = require_only_key(input_shapes, TensorSlotName::INPUT).require_variadic();

        return {
          {
            TensorSlotName::OUTPUT,
            SingularOrVariadic<TensorShape>{
              throw_if_unexpected(get_output_shape(attrs, inputs)),
            },
          },
        };
      },
      [&](Conv2DAttrs const &attrs) -> std::unordered_map<TensorSlotName, SingularOrVariadic<TensorShape>> {
        TensorShape input = require_only_key(input_shapes, TensorSlotName::INPUT).require_singular();

        return {
          {
            TensorSlotName::OUTPUT, 
            SingularOrVariadic<TensorShape>{
              get_output_shape(attrs, input),
            },
          },
        };
      },
      [&](DropoutAttrs const &attrs) -> std::unordered_map<TensorSlotName, SingularOrVariadic<TensorShape>> {
        TensorShape input = require_only_key(input_shapes, TensorSlotName::INPUT).require_singular();

        return {
          {
            TensorSlotName::OUTPUT,
            SingularOrVariadic<TensorShape>{
              get_output_shape(attrs, input),
            },
          },
        };
      },
      [&](ElementBinaryAttrs const &attrs) -> std::unordered_map<TensorSlotName, SingularOrVariadic<TensorShape>> {
        auto [lhs, rhs] = require_two_keys(input_shapes, TensorSlotName::LHS_INPUT, TensorSlotName::RHS_INPUT);

        return {
          {
            TensorSlotName::OUTPUT,
            SingularOrVariadic<TensorShape>{
              get_output_shape(attrs, lhs.require_singular(), rhs.require_singular()),
            },
          },
        };
      },
      [&](ElementUnaryAttrs const &attrs) -> std::unordered_map<TensorSlotName, SingularOrVariadic<TensorShape>> {
        TensorShape input = require_only_key(input_shapes, TensorSlotName::INPUT).require_singular();

        return {
          {
            TensorSlotName::OUTPUT,
            SingularOrVariadic<TensorShape>{
              get_output_shape(attrs, input),
            },
          },
        };
      },
      [&](EmbeddingAttrs const &attrs) -> std::unordered_map<TensorSlotName, SingularOrVariadic<TensorShape>> {
        TensorShape input = require_only_key(input_shapes, TensorSlotName::INPUT).require_singular();

        return {
          {
            TensorSlotName::OUTPUT,
            SingularOrVariadic<TensorShape>{
              throw_if_unexpected(get_output_shape(attrs, input)),
            },
          },
        };
      },
      [&](FlatAttrs const &attrs) -> std::unordered_map<TensorSlotName, SingularOrVariadic<TensorShape>> {
        TensorShape input = require_only_key(input_shapes, TensorSlotName::INPUT).require_singular();

        return {
          {
            TensorSlotName::OUTPUT,
            SingularOrVariadic<TensorShape>{
              get_output_shape(attrs, input),
            },
          },
        };
      },
      [&](GatherAttrs const &attrs) -> std::unordered_map<TensorSlotName, SingularOrVariadic<TensorShape>> {
        auto [input, index] = require_two_keys(input_shapes, TensorSlotName::INPUT, TensorSlotName::INDEX);

        return {
          {
            TensorSlotName::OUTPUT,
            SingularOrVariadic<TensorShape>{
              get_output_shape(attrs, input.require_singular(), index.require_singular()),
            },
          },
        };
      },
      [&](InputAttrs const &attrs) -> std::unordered_map<TensorSlotName, SingularOrVariadic<TensorShape>> {
        TensorShape input = require_only_key(input_shapes, TensorSlotName::INPUT).require_singular();

        return {
          {
            TensorSlotName::OUTPUT,
            SingularOrVariadic<TensorShape>{
              get_output_shape(attrs),
            },
          },
        };
      },
      [&](LayerNormAttrs const &attrs) -> std::unordered_map<TensorSlotName, SingularOrVariadic<TensorShape>> {
        TensorShape input = require_only_key(input_shapes, TensorSlotName::INPUT).require_singular();

        return {
          {
            TensorSlotName::OUTPUT,
            SingularOrVariadic<TensorShape>{
              throw_if_unexpected(get_output_shape(attrs, input)),
            },
          },
        };
      },
      [&](LinearAttrs const &attrs) -> std::unordered_map<TensorSlotName, SingularOrVariadic<TensorShape>> {
        TensorShape input = require_only_key(input_shapes, TensorSlotName::INPUT).require_singular();

        return {
          {
            TensorSlotName::OUTPUT,
            SingularOrVariadic<TensorShape>{
              throw_if_unexpected(get_output_shape(attrs, input)),
            },
          },
        };
      },
      [&](MultiHeadAttentionAttrs const &attrs) -> std::unordered_map<TensorSlotName, SingularOrVariadic<TensorShape>> {
        auto [query, key, value] = require_3(input_shapes, TensorSlotName::QUERY, TensorSlotName::KEY, TensorSlotName::VALUE);

        return {
          {
            TensorSlotName::OUTPUT,
            SingularOrVariadic<TensorShape>{
              throw_if_unexpected(get_output_shape(attrs, query.require_singular(), key.require_singular(), value.require_singular()))
            },
          },
        };
      },
      [&](Pool2DAttrs const &attrs) -> std::unordered_map<TensorSlotName, SingularOrVariadic<TensorShape>> {
        TensorShape input = require_only_key(input_shapes, TensorSlotName::INPUT).require_singular();
        
        return {
          {
            TensorSlotName::OUTPUT,
            SingularOrVariadic<TensorShape>{
              throw_if_unexpected(get_output_shape(attrs, input))
            },
          },
        };
      },
      [&](SoftmaxAttrs const &attrs) -> std::unordered_map<TensorSlotName, SingularOrVariadic<TensorShape>> {
        TensorShape input = require_only_key(input_shapes, TensorSlotName::INPUT).require_singular();

        return {
          {
            TensorSlotName::OUTPUT,
            SingularOrVariadic<TensorShape>{
              throw_if_unexpected(get_output_shape(attrs, input))
            },
          },
        };
      },
      [&](TransposeAttrs const &attrs) -> std::unordered_map<TensorSlotName, SingularOrVariadic<TensorShape>> {
        TensorShape input = require_only_key(input_shapes, TensorSlotName::INPUT).require_singular();

        return {
          {
            TensorSlotName::OUTPUT,
            SingularOrVariadic<TensorShape>{
              get_output_shape(attrs, input),
            },
          },
        };
      },
      [&](WeightAttrs const &attrs) -> std::unordered_map<TensorSlotName, SingularOrVariadic<TensorShape>> {
        TensorShape input = require_only_key(input_shapes, TensorSlotName::INPUT).require_singular();

        return {
          {
            TensorSlotName::OUTPUT,
            SingularOrVariadic<TensorShape>{
              get_output_shape(attrs),
            },
          },
        };
      },
      [&](auto const &attrs) -> std::unordered_map<TensorSlotName, SingularOrVariadic<TensorShape>> {
        NOT_IMPLEMENTED();
      }});
}

std::unordered_map<TensorSlotName, SingularOrVariadic<TensorShape>>
    get_weight_shapes(ComputationGraphOpAttrs const &op_attrs,
                      std::unordered_map<TensorSlotName, SingularOrVariadic<TensorShape>> const &input_shapes) {
  return op_attrs.visit<std::unordered_map<TensorSlotName, SingularOrVariadic<TensorShape>>>(overload{
      [&](BatchMatmulAttrs const &attrs) -> std::unordered_map<TensorSlotName, SingularOrVariadic<TensorShape>> {
        require_two_keys(input_shapes, TensorSlotName::LHS_INPUT, TensorSlotName::RHS_INPUT);

        return {};
      },
      [&](BatchNormAttrs const &attrs) -> std::unordered_map<TensorSlotName, SingularOrVariadic<TensorShape>> {
        TensorShape input = require_only_key(input_shapes, TensorSlotName::INPUT).require_singular();

        return throw_if_unexpected(
            get_weight_shapes(attrs, input));
      },
      [&](CastAttrs const &attrs) -> std::unordered_map<TensorSlotName, SingularOrVariadic<TensorShape>> { 
        require_only_key(input_shapes, TensorSlotName::INPUT);
        return {}; 
      },
      [&](ConcatAttrs const &attrs) -> std::unordered_map<TensorSlotName, SingularOrVariadic<TensorShape>> { 
        require_only_key(input_shapes, TensorSlotName::INPUT);

        return {}; 
      },
      [&](Conv2DAttrs const &attrs) -> std::unordered_map<TensorSlotName, SingularOrVariadic<TensorShape>> {
        TensorShape input = require_only_key(input_shapes, TensorSlotName::INPUT).require_singular();

        return get_weight_shapes(attrs, input);
      },
      [&](DropoutAttrs const &attrs) -> std::unordered_map<TensorSlotName, SingularOrVariadic<TensorShape>> { 
        return {}; 
      },
      [&](ElementBinaryAttrs const &attrs) -> std::unordered_map<TensorSlotName, SingularOrVariadic<TensorShape>> {
        return {};
      },
      [&](ElementUnaryAttrs const &attrs) -> std::unordered_map<TensorSlotName, SingularOrVariadic<TensorShape>> {
        return {};
      },
      [&](EmbeddingAttrs const &attrs) -> std::unordered_map<TensorSlotName, SingularOrVariadic<TensorShape>> {
        TensorShape input = require_only_key(input_shapes, TensorSlotName::INPUT).require_singular();

        return {
          {
            TensorSlotName::WEIGHT,
            SingularOrVariadic<TensorShape>{
              throw_if_unexpected(get_weights_shape(attrs, input)),
            },
          },
        };
      },
      [&](FlatAttrs const &attrs) -> std::unordered_map<TensorSlotName, SingularOrVariadic<TensorShape>> { 
        require_only_key(input_shapes, TensorSlotName::INPUT).require_singular();
        return {}; 
      },
      [&](GatherAttrs const &attrs) -> std::unordered_map<TensorSlotName, SingularOrVariadic<TensorShape>> {
        require_two_keys(input_shapes, TensorSlotName::INPUT, TensorSlotName::INDEX);
        return {}; 
      },
      [&](InputAttrs const &attrs) -> std::unordered_map<TensorSlotName, SingularOrVariadic<TensorShape>> { 
        ASSERT(input_shapes.size() == 0);
        return {}; 
      },
      [&](LayerNormAttrs const &attrs) -> std::unordered_map<TensorSlotName, SingularOrVariadic<TensorShape>> {
        TensorShape input = require_only_key(input_shapes, TensorSlotName::INPUT).require_singular();

        return throw_if_unexpected(get_weight_shapes(attrs, input));
      },
      [&](LinearAttrs const &attrs) -> std::unordered_map<TensorSlotName, SingularOrVariadic<TensorShape>> {
        TensorShape input = require_only_key(input_shapes, TensorSlotName::INPUT).require_singular();

        return throw_if_unexpected(get_weight_shapes(attrs, input));
      },
      [&](MultiHeadAttentionAttrs const &attrs) -> std::unordered_map<TensorSlotName, SingularOrVariadic<TensorShape>> {
        auto [query, key, value] = require_3(input_shapes, TensorSlotName::QUERY, TensorSlotName::KEY, TensorSlotName::VALUE);

        return throw_if_unexpected(get_weight_shapes(attrs, query.require_singular(), key.require_singular(), value.require_singular()));
      },
      [&](Pool2DAttrs const &attrs) -> std::unordered_map<TensorSlotName, SingularOrVariadic<TensorShape>> { 
        require_only_key(input_shapes, TensorSlotName::INPUT).require_singular();

        return {}; 
      },
      [&](SoftmaxAttrs const &attrs) -> std::unordered_map<TensorSlotName, SingularOrVariadic<TensorShape>> { 
        require_only_key(input_shapes, TensorSlotName::INPUT).require_singular();

        return {}; 
      },
      [&](WeightAttrs const &attrs) -> std::unordered_map<TensorSlotName, SingularOrVariadic<TensorShape>> { 
        ASSERT(input_shapes.size() == 0);
        return {}; 
      },
      [&](auto const &attrs) -> std::unordered_map<TensorSlotName, SingularOrVariadic<TensorShape>> {
        NOT_IMPLEMENTED();
      }});
}

std::unordered_map<TensorSlotName, SingularOrVariadic<ParallelTensorShape>>
    get_output_shapes(PCGOperatorAttrs const &pcg_op_attrs,
                      std::unordered_map<TensorSlotName, SingularOrVariadic<ParallelTensorShape>> const &input_shapes) {
  return pcg_op_attrs.visit<std::unordered_map<TensorSlotName, SingularOrVariadic<ParallelTensorShape>>>(overload{
      [&](BatchMatmulAttrs const &attrs) -> std::unordered_map<TensorSlotName, SingularOrVariadic<ParallelTensorShape>> {
        auto [lhs, rhs] = require_two_keys(input_shapes, TensorSlotName::LHS_INPUT, TensorSlotName::RHS_INPUT);

        return {
          {
            TensorSlotName::OUTPUT,
            SingularOrVariadic<ParallelTensorShape>{
              throw_if_unexpected(get_output_shape(attrs, lhs.require_singular(), rhs.require_singular())),
            },
          },
        };
      },
      [&](BatchNormAttrs const &attrs) -> std::unordered_map<TensorSlotName, SingularOrVariadic<ParallelTensorShape>> {
        ParallelTensorShape input = require_only_key(input_shapes, TensorSlotName::INPUT).require_singular();

        return {
          {
            TensorSlotName::OUTPUT,
            SingularOrVariadic<ParallelTensorShape>{
              throw_if_unexpected(get_output_shape(attrs, input)),
            },
          },
        };
      },
      [&](CastAttrs const &attrs) -> std::unordered_map<TensorSlotName, SingularOrVariadic<ParallelTensorShape>> {
        ParallelTensorShape input = require_only_key(input_shapes, TensorSlotName::INPUT).require_singular();

        return {
          {
            TensorSlotName::OUTPUT,
            SingularOrVariadic<ParallelTensorShape>{
              throw_if_unexpected(get_output_shape(attrs, input))
            },
          },
        };
      },
      [&](CombineAttrs const &attrs) -> std::unordered_map<TensorSlotName, SingularOrVariadic<ParallelTensorShape>> {
        ParallelTensorShape input = require_only_key(input_shapes, TensorSlotName::INPUT).require_singular();

        return {
          {
            TensorSlotName::OUTPUT,
            SingularOrVariadic<ParallelTensorShape>{
              throw_if_unexpected(get_output_shape(attrs, input))
            },
          },
        };
      },
      [&](ConcatAttrs const &attrs) -> std::unordered_map<TensorSlotName, SingularOrVariadic<ParallelTensorShape>> {
        std::vector<ParallelTensorShape> inputs = require_only_key(input_shapes, TensorSlotName::INPUT).require_variadic();

        return {
          {
            TensorSlotName::OUTPUT,
            SingularOrVariadic<ParallelTensorShape>{
              throw_if_unexpected(get_output_shape(attrs, inputs))
            },
          },
        };
      },
      [&](Conv2DAttrs const &attrs) -> std::unordered_map<TensorSlotName, SingularOrVariadic<ParallelTensorShape>> {
        ParallelTensorShape input = require_only_key(input_shapes, TensorSlotName::INPUT).require_singular();

        return {
          {
            TensorSlotName::OUTPUT,
            SingularOrVariadic<ParallelTensorShape>{
              get_output_shape(attrs, input)
            },
          },
        };
      },
      [&](DropoutAttrs const &attrs) -> std::unordered_map<TensorSlotName, SingularOrVariadic<ParallelTensorShape>> {
        ParallelTensorShape input = require_only_key(input_shapes, TensorSlotName::INPUT).require_singular();

        return {
          {
            TensorSlotName::OUTPUT,
            SingularOrVariadic<ParallelTensorShape>{
              throw_if_unexpected(get_output_shape(attrs, input)),
            },
          },
        };
      },
      [&](ElementBinaryAttrs const &attrs) -> std::unordered_map<TensorSlotName, SingularOrVariadic<ParallelTensorShape>> {
        auto [lhs, rhs] = require_two_keys(input_shapes, TensorSlotName::LHS_INPUT, TensorSlotName::RHS_INPUT);

        return {
          {
            TensorSlotName::OUTPUT,
            SingularOrVariadic<ParallelTensorShape>{
              get_output_shape(attrs, lhs.require_singular(), rhs.require_singular()),
            },
          },
        };
      },
      [&](ElementUnaryAttrs const &attrs) -> std::unordered_map<TensorSlotName, SingularOrVariadic<ParallelTensorShape>> {
        ParallelTensorShape input = require_only_key(input_shapes, TensorSlotName::INPUT).require_singular();

        return {
          {
            TensorSlotName::OUTPUT,
            SingularOrVariadic<ParallelTensorShape>{
              get_output_shape(attrs, input),
            },
          },
        };
      },
      [&](EmbeddingAttrs const &attrs) -> std::unordered_map<TensorSlotName, SingularOrVariadic<ParallelTensorShape>> {
        ParallelTensorShape input = require_only_key(input_shapes, TensorSlotName::INPUT).require_singular();

        return {
          {
            TensorSlotName::OUTPUT,
            SingularOrVariadic<ParallelTensorShape>{
              throw_if_unexpected(get_output_shape(attrs, input)),
            },
          },
        };
      },
      [&](FlatAttrs const &attrs) -> std::unordered_map<TensorSlotName, SingularOrVariadic<ParallelTensorShape>> {
        ParallelTensorShape input = require_only_key(input_shapes, TensorSlotName::INPUT).require_singular();

        return {
          {
            TensorSlotName::OUTPUT,
            SingularOrVariadic<ParallelTensorShape>{
              get_output_shape(attrs, input),
            },
          },
        };
      },
      [&](GatherAttrs const &attrs) -> std::unordered_map<TensorSlotName, SingularOrVariadic<ParallelTensorShape>> {
        auto [input, index] = require_two_keys(input_shapes, TensorSlotName::INPUT, TensorSlotName::INDEX);

        return {
          {
            TensorSlotName::OUTPUT,
            SingularOrVariadic<ParallelTensorShape>{
              get_output_shape(attrs, input.require_singular(), index.require_singular()),
            },
          },
        };
      },
      [&](InputAttrs const &attrs) -> std::unordered_map<TensorSlotName, SingularOrVariadic<ParallelTensorShape>> {
        ASSERT(input_shapes.size() == 0);

        return {
          {
            TensorSlotName::OUTPUT,
            SingularOrVariadic<ParallelTensorShape>{
              get_output_parallel_tensor_shape(attrs),
            },
          },
        };
      },
      [&](LayerNormAttrs const &attrs) -> std::unordered_map<TensorSlotName, SingularOrVariadic<ParallelTensorShape>> {
        ParallelTensorShape input = require_only_key(input_shapes, TensorSlotName::INPUT).require_singular();

        return {
          {
            TensorSlotName::OUTPUT,
            SingularOrVariadic<ParallelTensorShape>{
              throw_if_unexpected(get_output_shape(attrs, input)),
            },
          },
        };
      },
      [&](LinearAttrs const &attrs) -> std::unordered_map<TensorSlotName, SingularOrVariadic<ParallelTensorShape>> {
        ParallelTensorShape input = require_only_key(input_shapes, TensorSlotName::INPUT).require_singular();

        return {
          {
            TensorSlotName::OUTPUT,
            SingularOrVariadic<ParallelTensorShape>{
              throw_if_unexpected(get_output_shape(attrs, input)),
            },
          },
        };
      },
      [&](MultiHeadAttentionAttrs const &attrs)
          -> std::unordered_map<TensorSlotName, SingularOrVariadic<ParallelTensorShape>> {
        auto [i1, i2, i3] = require_3(input_shapes, TensorSlotName::QUERY, TensorSlotName::KEY, TensorSlotName::VALUE);

        return {
          {
            TensorSlotName::OUTPUT,
            SingularOrVariadic<ParallelTensorShape>{
              throw_if_unexpected(get_output_shape(attrs, i1.require_singular(), i2.require_singular(), i3.require_singular()))
            },
          },
        };
      },
      [&](Pool2DAttrs const &attrs) -> std::unordered_map<TensorSlotName, SingularOrVariadic<ParallelTensorShape>> {
        ParallelTensorShape input = require_only_key(input_shapes, TensorSlotName::INPUT).require_singular();

        return {
          {
            TensorSlotName::OUTPUT,
            SingularOrVariadic<ParallelTensorShape>{
              throw_if_unexpected(get_output_shape(attrs, input)),
            },
          },
        };
      },
      [&](ReductionAttrs const &attrs) -> std::unordered_map<TensorSlotName, SingularOrVariadic<ParallelTensorShape>> {
        ParallelTensorShape input = require_only_key(input_shapes, TensorSlotName::INPUT).require_singular();

        return {
          {
            TensorSlotName::OUTPUT,
            SingularOrVariadic<ParallelTensorShape>{
              throw_if_unexpected(get_output_shape(attrs, input)),
            },
          },
        };
      },
      [&](RepartitionAttrs const &attrs) -> std::unordered_map<TensorSlotName, SingularOrVariadic<ParallelTensorShape>> {
        ParallelTensorShape input = require_only_key(input_shapes, TensorSlotName::INPUT).require_singular();

        return {
          {
            TensorSlotName::OUTPUT,
            SingularOrVariadic<ParallelTensorShape>{
              throw_if_unexpected(get_output_shape(attrs, input)),
            },
          },
        };
      },
      [&](ReplicateAttrs const &attrs) -> std::unordered_map<TensorSlotName, SingularOrVariadic<ParallelTensorShape>> {
        ParallelTensorShape input = require_only_key(input_shapes, TensorSlotName::INPUT).require_singular();

        return {
          {
            TensorSlotName::OUTPUT,
            SingularOrVariadic<ParallelTensorShape>{
              get_output_shape(attrs, input),
            },
          },
        };
      },
      [&](SoftmaxAttrs const &attrs) -> std::unordered_map<TensorSlotName, SingularOrVariadic<ParallelTensorShape>> {
        ParallelTensorShape input = require_only_key(input_shapes, TensorSlotName::INPUT).require_singular();

        return {
          {
            TensorSlotName::OUTPUT,
            SingularOrVariadic<ParallelTensorShape>{
              throw_if_unexpected(get_output_shape(attrs, input)),
            },
          },
        };
      },
      [&](TransposeAttrs const &attrs) -> std::unordered_map<TensorSlotName, SingularOrVariadic<ParallelTensorShape>> {
        ParallelTensorShape input = require_only_key(input_shapes, TensorSlotName::INPUT).require_singular();

        return {
          {
            TensorSlotName::OUTPUT,
            SingularOrVariadic<ParallelTensorShape>{
              get_output_shape(attrs, input),
            },
          },
        };
      },
      [&](WeightAttrs const &attrs) -> std::unordered_map<TensorSlotName, SingularOrVariadic<ParallelTensorShape>> {
        ASSERT(input_shapes.size() == 0);

        return {
          {
            TensorSlotName::OUTPUT,
            SingularOrVariadic<ParallelTensorShape>{
              get_output_parallel_tensor_shape(attrs),
            },
          },
        };
      },
      [&](auto const &attrs) -> std::unordered_map<TensorSlotName, SingularOrVariadic<ParallelTensorShape>> {
        NOT_IMPLEMENTED();
      }});
}

std::unordered_map<TensorSlotName, SingularOrVariadic<ParallelTensorShape>>
    get_weight_shapes(PCGOperatorAttrs const &pcg_op_attrs,
                      std::unordered_map<TensorSlotName, SingularOrVariadic<ParallelTensorShape>> const &input_shapes) {
  return pcg_op_attrs.visit<std::unordered_map<TensorSlotName, SingularOrVariadic<ParallelTensorShape>>>(overload{
      [&](BatchMatmulAttrs const &attrs) -> std::unordered_map<TensorSlotName, SingularOrVariadic<ParallelTensorShape>> {
        require_two_keys(input_shapes, TensorSlotName::LHS_INPUT, TensorSlotName::RHS_INPUT);

        return {};
      },
      [&](BatchNormAttrs const &attrs) -> std::unordered_map<TensorSlotName, SingularOrVariadic<ParallelTensorShape>> {
        ParallelTensorShape input = require_only_key(input_shapes, TensorSlotName::INPUT).require_singular();
        
        return throw_if_unexpected(get_weight_shapes(attrs, input));
      },
      [&](CastAttrs const &attrs) -> std::unordered_map<TensorSlotName, SingularOrVariadic<ParallelTensorShape>> {
        require_only_key(input_shapes, TensorSlotName::INPUT).require_singular();

        return {};
      },
      [&](CombineAttrs const &attrs) -> std::unordered_map<TensorSlotName, SingularOrVariadic<ParallelTensorShape>> {
        require_only_key(input_shapes, TensorSlotName::INPUT).require_singular();

        return {};
      },
      [&](ConcatAttrs const &attrs) -> std::unordered_map<TensorSlotName, SingularOrVariadic<ParallelTensorShape>> {
        require_only_key(input_shapes, TensorSlotName::INPUT).require_variadic();

        return {};
      },
      [&](Conv2DAttrs const &attrs) -> std::unordered_map<TensorSlotName, SingularOrVariadic<ParallelTensorShape>> {
        ParallelTensorShape input = require_only_key(input_shapes, TensorSlotName::INPUT).require_singular();

        return get_weight_shapes(attrs, input);
      },
      [&](DropoutAttrs const &attrs) -> std::unordered_map<TensorSlotName, SingularOrVariadic<ParallelTensorShape>> {
        require_only_key(input_shapes, TensorSlotName::INPUT).require_singular();

        return {};
      },
      [&](ElementBinaryAttrs const &attrs) -> std::unordered_map<TensorSlotName, SingularOrVariadic<ParallelTensorShape>> {
        require_two_keys(input_shapes, TensorSlotName::LHS_INPUT, TensorSlotName::RHS_INPUT);

        return {};
      },
      [&](ElementUnaryAttrs const &attrs) -> std::unordered_map<TensorSlotName, SingularOrVariadic<ParallelTensorShape>> {
        require_only_key(input_shapes, TensorSlotName::INPUT).require_singular();

        return {};
      },
      [&](EmbeddingAttrs const &attrs) -> std::unordered_map<TensorSlotName, SingularOrVariadic<ParallelTensorShape>> {
        ParallelTensorShape input = require_only_key(input_shapes, TensorSlotName::INPUT).require_singular();

        return {
          {
            TensorSlotName::WEIGHT,
            SingularOrVariadic<ParallelTensorShape>{
              throw_if_unexpected(get_weights_shape(attrs, input)),
            },
          },
        };
      },
      [&](FlatAttrs const &attrs) -> std::unordered_map<TensorSlotName, SingularOrVariadic<ParallelTensorShape>> {
        require_only_key(input_shapes, TensorSlotName::INPUT).require_singular();

        return {};
      },
      [&](GatherAttrs const &attrs) -> std::unordered_map<TensorSlotName, SingularOrVariadic<ParallelTensorShape>> {
        require_two_keys(input_shapes, TensorSlotName::INPUT, TensorSlotName::INDEX);

        return {};
      },
      [&](InputAttrs const &attrs) -> std::unordered_map<TensorSlotName, SingularOrVariadic<ParallelTensorShape>> {
        ASSERT(input_shapes.size() == 0);

        return {};
      },
      [&](LayerNormAttrs const &attrs) -> std::unordered_map<TensorSlotName, SingularOrVariadic<ParallelTensorShape>> {
        ParallelTensorShape input = require_only_key(input_shapes, TensorSlotName::INPUT).require_singular();

        return throw_if_unexpected(get_weight_shapes(attrs, input));
      },
      [&](LinearAttrs const &attrs) -> std::unordered_map<TensorSlotName, SingularOrVariadic<ParallelTensorShape>> {
        ParallelTensorShape input = require_only_key(input_shapes, TensorSlotName::INPUT).require_singular();

        return throw_if_unexpected(get_weight_shapes(attrs, input));
      },
      [&](MultiHeadAttentionAttrs const &attrs)
          -> std::unordered_map<TensorSlotName, SingularOrVariadic<ParallelTensorShape>> {
        auto [query, key, value] = require_3(input_shapes, TensorSlotName::QUERY, TensorSlotName::KEY, TensorSlotName::VALUE);

        return throw_if_unexpected(get_weight_shapes(attrs, query.require_singular(), key.require_singular(), value.require_singular()));
      },
      [&](Pool2DAttrs const &attrs) -> std::unordered_map<TensorSlotName, SingularOrVariadic<ParallelTensorShape>> {
        require_only_key(input_shapes, TensorSlotName::INPUT).require_singular();

        return {};
      },
      [&](RepartitionAttrs const &attrs) -> std::unordered_map<TensorSlotName, SingularOrVariadic<ParallelTensorShape>> {
        require_only_key(input_shapes, TensorSlotName::INPUT).require_singular();

        return {};
      },
      [&](ReplicateAttrs const &attrs) -> std::unordered_map<TensorSlotName, SingularOrVariadic<ParallelTensorShape>> {
        require_only_key(input_shapes, TensorSlotName::INPUT).require_singular();

        return {};
      },
      [&](ReductionAttrs const &attrs) -> std::unordered_map<TensorSlotName, SingularOrVariadic<ParallelTensorShape>> {
        require_only_key(input_shapes, TensorSlotName::INPUT).require_singular();

        return {};
      },
      [&](SoftmaxAttrs const &attrs) -> std::unordered_map<TensorSlotName, SingularOrVariadic<ParallelTensorShape>> {
        require_only_key(input_shapes, TensorSlotName::INPUT).require_singular();

        return {};
      },
      [&](TransposeAttrs const &attrs) -> std::unordered_map<TensorSlotName, SingularOrVariadic<ParallelTensorShape>> {
        require_only_key(input_shapes, TensorSlotName::INPUT).require_singular();

        return {};
      },
      [&](WeightAttrs const &attrs) -> std::unordered_map<TensorSlotName, SingularOrVariadic<ParallelTensorShape>> {
        require_only_key(input_shapes, TensorSlotName::INPUT).require_singular();

        return {};
      },
      [&](auto const &attrs) -> std::unordered_map<TensorSlotName, SingularOrVariadic<ParallelTensorShape>> {
        NOT_IMPLEMENTED();
      }});
}

} // namespace FlexFlow
