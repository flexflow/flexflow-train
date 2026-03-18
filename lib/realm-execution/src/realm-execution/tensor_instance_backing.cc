#include "realm-execution/tensor_instance_backing.h"
#include "utils/containers/values.h"

namespace FlexFlow {

TensorInstanceBacking make_empty_tensor_instance_backing() {
  return TensorInstanceBacking{
      /*backing=*/{},
  };
}

TensorInstanceBacking subset_tensor_instance_backing_for_invocation(
    TensorInstanceBacking const &backing,
    DynamicNodeInvocation const &invocation) {
  TensorInstanceBacking result = make_empty_tensor_instance_backing();
  for (DynamicValueAttrs const &value : values(invocation.inputs)) {
    result.backing.insert(std::pair{value, backing.backing.at(value)});
  }
  for (DynamicValueAttrs const &value : values(invocation.outputs)) {
    result.backing.insert(std::pair{value, backing.backing.at(value)});
  }
  return result;
}

} // namespace FlexFlow
