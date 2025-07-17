#ifndef _FLEXFLOW_LIB_KERNELS_INCLUDE_KERNELS_FORMAT_ACCESSOR_CONTENTS_H
#define _FLEXFLOW_LIB_KERNELS_INCLUDE_KERNELS_FORMAT_ACCESSOR_CONTENTS_H

#include "kernels/accessor.h"

namespace FlexFlow {

std::string format_accessor_r_contents(GenericTensorAccessorR const &);
std::string format_accessor_w_contents(GenericTensorAccessorW const &);

} // namespace FlexFlow

#endif
