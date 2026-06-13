#ifndef _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_GLOBAL_DEVICE_ID_T_H
#define _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_GLOBAL_DEVICE_ID_T_H

#include "task-spec/global_device_id_t.dtg.h"
#include "task-spec/local_device_id_t.dtg.h"
#include "pcg/node_idx_t.dtg.h"

namespace FlexFlow {

global_device_id_t global_device_id_from_local(local_device_id_t const &, node_idx_t);
local_device_id_t local_device_id_from_global(global_device_id_t const &);

} // namespace FlexFlow

#endif
