#ifndef _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_RELATIVE_FF_DIM_T_H
#define _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_RELATIVE_FF_DIM_T_H

#include "op-attrs/ff_dim_t.dtg.h"
#include "op-attrs/relative_ff_dim_t.dtg.h"
#include "rapidcheck.h"

namespace FlexFlow {
ff_dim_t relative_ff_dim_t_to_ff_dim_t(relative_ff_dim_t ff_dim, int input_dim);
} // namespace FlexFlow

namespace rc {
template <>
struct Arbitrary<::FlexFlow::relative_ff_dim_t> {
  static Gen<::FlexFlow::relative_ff_dim_t> arbitrary();
};
} // namespace rc

#endif // _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_RELATIVE_FF_DIM_T_H
