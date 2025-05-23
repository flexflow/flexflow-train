#ifndef _FLEXFLOW_LIB_KERNELS_INCLUDE_KERNELS_TENSOR_ACCESSOR_REDUCTIONS_H
#define _FLEXFLOW_LIB_KERNELS_INCLUDE_KERNELS_TENSOR_ACCESSOR_REDUCTIONS_H

#include "kernels/accessor.h"

namespace FlexFlow {

bool tensor_accessor_all(GenericTensorAccessorR const &);
bool tensor_accessor_any(GenericTensorAccessorR const &);

} // namespace FlexFlow

#endif
