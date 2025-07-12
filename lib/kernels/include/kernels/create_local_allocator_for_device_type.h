#ifndef _FLEXFLOW_LIB_KERNELS_INCLUDE_KERNELS_ALLOCATOR_FOR_DEVICE_TYPE_H
#define _FLEXFLOW_LIB_KERNELS_INCLUDE_KERNELS_ALLOCATOR_FOR_DEVICE_TYPE_H

#include "kernels/allocation.h"

namespace FlexFlow {

Allocator create_local_allocator_for_device_type(DeviceType);

} // namespace FlexFlow

#endif
