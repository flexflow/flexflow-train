#include "task-spec/training_tensor_group_with_attrs.h"

namespace FlexFlow {

TrainingTensorGroupWithAttrs
  make_training_tensor_group_with_attrs_from_group_and_attrs(
    TrainingTensorGroup const &group,
    TensorAttrs const &attrs) {

  return TrainingTensorGroupWithAttrs{
    /*tensor_attrs=*/attrs,
    /*forward_tensor=*/group.forward_tensor,
    /*gradient_tensor=*/group.gradient_tensor,
    /*optimizer_tensors=*/group.optimizer_tensors,
  };
}

TrainingTensorGroup tensor_group_without_attrs(TrainingTensorGroupWithAttrs const &with_attrs) {
  return TrainingTensorGroup{
    /*forward_tensor=*/with_attrs.forward_tensor,
    /*gradient_tensor=*/with_attrs.gradient_tensor,
    /*optimizer_tensors=*/with_attrs.optimizer_tensors,
  };
}

} // namespace FlexFlow
