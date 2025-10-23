#ifndef _FLEXFLOW_NOOP_H
#define _FLEXFLOW_NOOP_H

#include "op-attrs/ops/noop_attrs.dtg.h"
#include "task-spec/op_task_invocation.h"

namespace FlexFlow {

std::unordered_set<task_id_t> get_task_ids(NoopAttrs const &);

} // namespace FlexFlow

#endif
