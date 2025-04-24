#ifndef _FLEXFLOW_OPS_KERNELS_REDUCTION_KERNELS_H
#define _FLEXFLOW_OPS_KERNELS_REDUCTION_KERNELS_H

#include "device.h"
#include "kernels/accessor.h"

namespace FlexFlow::Kernels::Reduction {

void forward_kernel(ffStream_t stream,
                    GenericTensorAccessorR const &input,
                    GenericTensorAccessorW const &output,
                    size_t num_replicas);

void backward_kernel(ffStream_t stream,
                     GenericTensorAccessorR const &output,
                     GenericTensorAccessorW const &input);

} // namespace FlexFlow::Kernels::Reduction

#endif // _FLEXFLOW_OPS_KERNELS_REDUCTION_KERNELS_H
