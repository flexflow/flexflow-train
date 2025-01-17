#ifndef _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_FF_DIM_T_H
#define _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_FF_DIM_T_H

#include "op-attrs/ff_dim_t.dtg.h"
#include "op-attrs/relative_ff_dim_t.dtg.h"
#include "rapidcheck.h"

namespace FlexFlow {

relative_ff_dim_t relative_ff_dim_t_from_ff_dim_t(ff_dim_t ff_dim);

std::set<ff_dim_t> ff_dim_range(int end);

} // namespace FlexFlow

namespace rc {
template <>
struct Arbitrary<::FlexFlow::ff_dim_t> {
  static Gen<::FlexFlow::ff_dim_t> arbitrary();
};
} // namespace rc

#endif // _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_FF_DIM_T_H
