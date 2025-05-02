#ifndef _FLEXFLOW_LIB_KERNELS_INCLUDE_KERNELS_REVERSE_KERNELS_PARAMS_H
#define _FLEXFLOW_LIB_KERNELS_INCLUDE_KERNELS_REVERSE_KERNELS_PARAMS_H

#include "kernels/reverse_kernels_params.dtg.h"
#include "op-attrs/ops/reverse_attrs.dtg.h"
#include "kernels/array_shape.h"

namespace FlexFlow {

ReverseKernelsParams compute_reverse_kernels_params
  (ArrayShape const &output_shape, ReverseAttrs const &attrs);

} // namespace FlexFlow

#endif
