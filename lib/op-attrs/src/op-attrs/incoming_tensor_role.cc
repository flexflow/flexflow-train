#include "op-attrs/incoming_tensor_role.h"
#include <libassert/assert.hpp>

namespace FlexFlow {

TensorRole
    tensor_role_from_incoming_tensor_role(IncomingTensorRole incoming_role) {
  switch (incoming_role) {
    case IncomingTensorRole::INPUT:
      return TensorRole::INPUT;
    case IncomingTensorRole::WEIGHT:
      return TensorRole::WEIGHT;
    default:
      PANIC(fmt::format("Unhandled IncomingTensorRole {}", incoming_role));
  }
}

} // namespace FlexFlow
