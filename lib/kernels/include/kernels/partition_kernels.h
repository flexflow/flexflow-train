#ifndef _FLEXFLOW_OPS_KERNELS_PARTITION_KERNELS_H
#define _FLEXFLOW_OPS_KERNELS_PARTITION_KERNELS_H

#include "kernels/accessor.h"
#include "kernels/device.h"

namespace FlexFlow {

struct RepartitionPerDeviceState {
  PerDeviceFFHandle handle;
  req<DataType> data_type;
};

FF_VISITABLE_STRUCT_NO_EQ(RepartitionPerDeviceState, handle, data_type);

namespace Kernels::Repartition {

RepartitionPerDeviceState init_kernel(PerDeviceFFHandle const &handle,
                                      DataType data_type);

void forward_kernel(ffStream_t stream,
                    RepartitionPerDeviceState const &m,
                    GenericTensorAccessorR const &input,
                    GenericTensorAccessorW const &output);

void backward_kernel(ffStream_t stream,
                     RepartitionPerDeviceState const &m,
                     GenericTensorAccessorR const &output_grad,
                     GenericTensorAccessorW const &input_grad);

} // namespace Kernels::Repartition
} // namespace FlexFlow

#endif // _FLEXFLOW_OPS_KERNELS_PARTITION_KERNELS_H
