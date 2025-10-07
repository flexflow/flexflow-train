#include "task-spec/symbolic_training_tensor_group_with_attrs.h"

namespace FlexFlow {

SymbolicTrainingTensorGroupWithAttrs
  make_symbolic_training_tensor_group_with_attrs_from_group_and_attrs(
    SymbolicTrainingTensorGroup const &tensor_group,
    TensorShape const &tensor_shape) {

  return SymbolicTrainingTensorGroupWithAttrs{
    /*tensor_shape=*/tensor_shape,
    /*forward_tensor=*/tensor_group.forward_tensor,
    /*gradient_tensor=*/tensor_group.gradient_tensor,
    /*optimizer_tensors=*/tensor_group.optimizer_tensors,
  };
}


} // namespace FlexFlow
