#ifndef _FLEXFLOW_PCG_OPTIMIZER_ATTRS_H
#define _FLEXFLOW_PCG_OPTIMIZER_ATTRS_H

#include "pcg/optimizer_attrs.dtg.h"
#include "pcg/optimizer_slot_name.dtg.h"
#include "utils/nonnegative_int/nonnegative_int.h"

namespace FlexFlow {

OptimizerAttrs get_optimizer_attrs_for_next_iter(OptimizerAttrs const &old);

std::unordered_set<OptimizerSlotName>
    get_slot_names_for_optimizer(OptimizerAttrs const &);

} // namespace FlexFlow

#endif
