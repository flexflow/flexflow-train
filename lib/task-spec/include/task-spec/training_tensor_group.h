#ifndef _FLEXFLOW_LIB_LOCAL_EXECUTION_INCLUDE_LOCAL_EXECUTION_TRAINING_TENSOR_GROUP_H
#define _FLEXFLOW_LIB_LOCAL_EXECUTION_INCLUDE_LOCAL_EXECUTION_TRAINING_TENSOR_GROUP_H

#include "pcg/optimizer_attrs.dtg.h"
#include "pcg/tensor_attrs.dtg.h"
#include "pcg/tensor_guid_t.dtg.h"
#include "task-spec/forward_tensor_source.h"
#include "task-spec/gradient_tensor_source.h"
#include "task-spec/optimizer_tensor_source.h"
#include "task-spec/training_tensor_group.dtg.h"
#include "task-spec/training_tensor_guid_t.dtg.h"

namespace FlexFlow {

TrainingTensorGroup 
  make_training_tensor_group_for_tensor_guid_t(tensor_guid_t tensor_guid,
                                               TensorAttrs const &tensor_attrs,
                                               OptimizerAttrs const &optimizer_attrs,
                                               ForwardTensorSource &forward_tensor_source,
                                               GradientTensorSource &gradient_tensor_source,
                                               OptimizerTensorSource &optimizer_tensor_source);

std::unordered_set<training_tensor_guid_t> get_all_training_tensors_in_tensor_group(TrainingTensorGroup const &);

} // namespace FlexFlow

#endif
