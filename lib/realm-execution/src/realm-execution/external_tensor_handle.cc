#include "realm-execution/external_tensor_handle.h"

namespace FlexFlow {

ExternalTensorHandle::ExternalTensorHandle(TensorShape const &shape,
                                           Realm::RegionInstance instance,
                                           Realm::Event ready,
                                           Allocator allocator,
                                           GenericTensorAccessorW accessor)
    : shape(shape), instance(instance), ready(ready), allocator(allocator),
      accessor(accessor) {}

float *ExternalTensorHandle::get_float_ptr() const {
  return this->accessor.get_float_ptr();
}

double *ExternalTensorHandle::get_double_ptr() const {
  return this->accessor.get_double_ptr();
}

void *ExternalTensorHandle::get_ptr() const {
  return this->accessor.ptr;
}

} // namespace FlexFlow
