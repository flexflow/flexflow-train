#include "task-spec/training_cg_op_attrs_and_signature_with_shapes.h"

namespace FlexFlow {

TrainingLayerSymbolicTensorGroupSignatureWithShapes
  get_signature_with_shapes(TrainingCgOpAttrsAndSignatureWithShapes const &attrs_and_signature) {

  return TrainingLayerSymbolicTensorGroupSignatureWithShapes{
    /*input_tensor_groups=*/attrs_and_signature.input_tensor_groups,
    /*weight_tensor_groups=*/attrs_and_signature.weight_tensor_groups,
    /*output_tensor_groups=*/attrs_and_signature.output_tensor_groups,
  };
}

TrainingCgOpAttrsAndSignatureWithShapes
  make_training_cg_op_attrs_and_signature(
    ComputationGraphOpAttrs const &op_attrs,
    TrainingLayerSymbolicTensorGroupSignatureWithShapes const &signature) {
  return TrainingCgOpAttrsAndSignatureWithShapes{
    /*op_attrs=*/op_attrs,
    /*input_tensor_groups=*/signature.input_tensor_groups,
    /*weight_tensor_groups=*/signature.weight_tensor_groups,
    /*output_tensor_groups=*/signature.output_tensor_groups,
  };
}

} // namespace FlexFlow
