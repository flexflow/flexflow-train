#ifndef _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_SYMBOLIC_TRAINING_TENSOR_GROUP_WITH_ATTRS_H
#define _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_SYMBOLIC_TRAINING_TENSOR_GROUP_WITH_ATTRS_H

#include "op-attrs/tensor_shape.dtg.h"
#include "task-spec/symbolic_training_tensor_group.dtg.h"
#include "task-spec/symbolic_training_tensor_group_with_attrs.dtg.h"

namespace FlexFlow {

SymbolicTrainingTensorGroupWithAttrs
  make_symbolic_training_tensor_group_with_attrs_from_group_and_attrs(
    SymbolicTrainingTensorGroup const &,
    TensorShape const &);

} // namespace FlexFlow

#endif
