#ifndef _FLEXFLOW_LIB_KERNELS_INCLUDE_KERNELS_REVERSE_KERNELS_PARAMS_H
#define _FLEXFLOW_LIB_KERNELS_INCLUDE_KERNELS_REVERSE_KERNELS_PARAMS_H

#include "kernels/reverse_kernels_params.dtg.h"
#include "op-attrs/ops/reverse_attrs.dtg.h"
#include "op-attrs/tensor_dims.dtg.h"

namespace FlexFlow {

ReverseKernelsParams
    compute_reverse_kernels_params(TensorDims const &output_dims,
                                   ReverseAttrs const &attrs);

} // namespace FlexFlow

#endif
