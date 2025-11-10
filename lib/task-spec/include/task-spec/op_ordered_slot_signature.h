#ifndef _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_OP_ORDERED_SLOT_SIGNATURE_H
#define _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_OP_ORDERED_SLOT_SIGNATURE_H

#include "task-spec/op_ordered_slot_signature.dtg.h"
#include "task-spec/ops/op_task_binding.h"

namespace FlexFlow {

OpOrderedSlotSignature get_op_ordered_slot_signature_for_binding(OpTaskBinding const &,
                                                                 nonnegative_int num_inputs,
                                                                 nonnegative_int num_weights,
                                                                 nonnegative_int num_outputs);

} // namespace FlexFlow

#endif
