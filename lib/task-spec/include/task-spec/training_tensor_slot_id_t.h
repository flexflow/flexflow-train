#ifndef _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_TRAINING_TENSOR_SLOT_ID_T_H
#define _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_TRAINING_TENSOR_SLOT_ID_T_H

#include "task-spec/fwb_tensor_slot_id_t.dtg.h"
#include "task-spec/training_tensor_slot_id_t.dtg.h"

namespace FlexFlow {

training_tensor_slot_id_t
  training_tensor_slot_from_fwb_slot(fwb_tensor_slot_id_t);

} // namespace FlexFlow

#endif
