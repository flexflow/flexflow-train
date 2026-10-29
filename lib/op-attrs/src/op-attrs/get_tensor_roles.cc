#include "op-attrs/get_tensor_roles.h"
#include "op-attrs/ops/linear.h"
#include "utils/overload.h"

namespace FlexFlow {

std::map<TensorSlotName, TensorRole>
    get_tensor_roles(PCGOperatorAttrs const &attrs) {
  return attrs.visit<std::map<TensorSlotName, TensorRole>>(overload{
      [&](ElementBinaryAttrs const &attrs)
          -> std::map<TensorSlotName, TensorRole> {
        return {
            {
                TensorSlotName::LHS_INPUT,
                TensorRole::INPUT,
            },
            {
                TensorSlotName::RHS_INPUT,
                TensorRole::INPUT,
            },
            {
                TensorSlotName::OUTPUT,
                TensorRole::OUTPUT,
            },
        };
      },
      [&](ElementUnaryAttrs const &attrs)
          -> std::map<TensorSlotName, TensorRole> {
        return {
            {
                TensorSlotName::INPUT,
                TensorRole::INPUT,
            },
            {
                TensorSlotName::OUTPUT,
                TensorRole::OUTPUT,
            },
        };
      },
      [&](InputAttrs const &attrs) -> std::map<TensorSlotName, TensorRole> {
        return {
            {
                TensorSlotName::OUTPUT,
                TensorRole::OUTPUT,
            },
        };
      },
      [&](LinearAttrs const &attrs) -> std::map<TensorSlotName, TensorRole> {
        return linear_get_tensor_roles(attrs);
      },
      [&](TransposeAttrs const &attrs) -> std::map<TensorSlotName, TensorRole> {
        return {
            {
                TensorSlotName::INPUT,
                TensorRole::INPUT,
            },
            {
                TensorSlotName::OUTPUT,
                TensorRole::OUTPUT,
            },
        };
      },
      [&](WeightAttrs const &attrs) -> std::map<TensorSlotName, TensorRole> {
        return {
            {
                TensorSlotName::OUTPUT,
                TensorRole::OUTPUT,
            },
        };
      },
      [&](auto const &attrs) -> std::map<TensorSlotName, TensorRole> {
        PANIC();
      },
  });
}

} // namespace FlexFlow
