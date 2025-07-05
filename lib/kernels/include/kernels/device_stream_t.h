#ifndef _FLEXFLOW_LIB_KERNELS_INCLUDE_KERNELS_DEVICE_STREAM_T_H
#define _FLEXFLOW_LIB_KERNELS_INCLUDE_KERNELS_DEVICE_STREAM_T_H

#include "kernels/device_stream_t.dtg.h"
#include "pcg/device_type.dtg.h"

namespace FlexFlow {

device_stream_t get_gpu_device_stream();
device_stream_t get_cpu_device_stream();
device_stream_t get_stream_for_device_type(DeviceType);

} // namespace FlexFlow

#endif
