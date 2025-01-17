#ifndef _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_TENSOR_NUM_DIMS_H
#define _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_TENSOR_NUM_DIMS_H

#include "op-attrs/ff_dim.dtg.h"
#include "op-attrs/tensor_num_dims.dtg.h"

namespace FlexFlow {

std::set<ff_dim_t> ff_dim_idxs_for_num_dims(TensorNumDims const &);

} // namespace FlexFlow

#endif
