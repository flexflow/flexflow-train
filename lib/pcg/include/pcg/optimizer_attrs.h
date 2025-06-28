#ifndef _FLEXFLOW_PCG_OPTIMIZER_ATTRS_H
#define _FLEXFLOW_PCG_OPTIMIZER_ATTRS_H

#include "pcg/optimizer_attrs.dtg.h"
#include "utils/nonnegative_int/nonnegative_int.h"

namespace FlexFlow {

OptimizerAttrs get_optimizer_attrs_for_next_iter(OptimizerAttrs const &old);
nonnegative_int get_num_optimizer_tensors(OptimizerAttrs const &);

} // namespace FlexFlow

#endif
