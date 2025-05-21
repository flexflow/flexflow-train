#ifndef _FLEXFLOW_LIB_KERNELS_INCLUDE_KERNELS_ARRAY_COORD_H
#define _FLEXFLOW_LIB_KERNELS_INCLUDE_KERNELS_ARRAY_COORD_H

#include "kernels/array_coord.dtg.h"

namespace FlexFlow {

ArrayCoord array_coord_drop_dims(ArrayCoord const &coord,
                                 std::function<bool(ff_dim_t)> const &should_drop_dim);

} // namespace FlexFlow

#endif
