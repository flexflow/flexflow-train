#ifndef _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_SYMBOLIC_TRAINING_TENSOR_GROUP_H
#define _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_SYMBOLIC_TRAINING_TENSOR_GROUP_H

#include "op-attrs/tensor_role.dtg.h"
#include "pcg/optimizer_attrs.dtg.h"
#include "pcg/tensor_attrs.dtg.h"
#include "pcg/tensor_guid_t.dtg.h"
#include "task-spec/op_training_tensor_type.dtg.h"
#include "task-spec/symbolic_forward_tensor_source.h"
#include "task-spec/symbolic_gradient_tensor_source.h"
#include "task-spec/symbolic_optimizer_tensor_source.h"
#include "task-spec/symbolic_training_tensor_group.dtg.h"
#include "task-spec/symbolic_training_tensor_guid_t.dtg.h"
#include "task-spec/fwb_tensor_type.dtg.h"

namespace FlexFlow {

SymbolicTrainingTensorGroup make_symbolic_training_tensor_group(
    CreateGrad create_grad,
    OptimizerAttrs const &optimizer_attrs,
    SymbolicForwardTensorSource &forward_tensor_source,
    SymbolicGradientTensorSource &gradient_tensor_source,
    SymbolicOptimizerTensorSource &optimizer_tensor_source);

symbolic_training_tensor_guid_t 
  get_training_tensor_for_type(SymbolicTrainingTensorGroup const &, FwbTensorType);

std::unordered_set<symbolic_training_tensor_guid_t>
    get_all_training_tensors_in_tensor_group(SymbolicTrainingTensorGroup const &);

} // namespace FlexFlow

#endif
