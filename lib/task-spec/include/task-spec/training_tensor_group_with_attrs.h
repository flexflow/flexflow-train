#ifndef _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_TRAINING_TENSOR_GROUP_WITH_ATTRS_H
#define _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_TRAINING_TENSOR_GROUP_WITH_ATTRS_H

#include "task-spec/training_tensor_group.dtg.h"
#include "task-spec/training_tensor_group_with_attrs.dtg.h"

namespace FlexFlow {

TrainingTensorGroupWithAttrs
  make_training_tensor_group_with_attrs_from_group_and_attrs(TrainingTensorGroup const &group,
                                             TensorAttrs const &attrs);

TrainingTensorGroup tensor_group_without_attrs(TrainingTensorGroupWithAttrs const &);

} // namespace FlexFlow

#endif
