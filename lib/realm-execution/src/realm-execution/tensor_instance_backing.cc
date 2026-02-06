#include "realm-execution/tensor_instance_backing.h"

namespace FlexFlow {

TensorInstanceBacking make_empty_tensor_instance_backing() {
  return TensorInstanceBacking{
      /*backing=*/{},
  };
}

} // namespace FlexFlow
