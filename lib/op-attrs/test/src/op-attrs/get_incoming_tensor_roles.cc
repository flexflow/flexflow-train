#include "op-attrs/get_incoming_tensor_roles.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE(
      "get_incoming_tensor_roles(ComputationGraphOpAttrs, int num_incoming)") {
    SUBCASE("Concat") {
      ComputationGraphOpAttrs attrs =
          ComputationGraphOpAttrs{ConcatAttrs{ff_dim_t{0_n}}};

      std::unordered_map<TensorSlotName, IncomingTensorRole> result =
          get_incoming_tensor_roles(attrs);
      std::unordered_map<TensorSlotName, IncomingTensorRole> correct = {
        {
          TensorSlotName::INPUT,
          IncomingTensorRole::INPUT,
        },
      };

      CHECK(result == correct);
    }
  }
}
