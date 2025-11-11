#include "task-spec/symbolic/symbolic_cg_op_attrs_and_training_signature_with_shapes.h"

namespace FlexFlow {

SymbolicLayerTrainingTensorGroupSignatureWithShapes
  get_signature_with_shapes(SymbolicCgOpAttrsAndTrainingSignatureWithShapes const &attrs_and_signature) {

  return SymbolicLayerTrainingTensorGroupSignatureWithShapes{
    /*input_tensor_groups=*/attrs_and_signature.input_tensor_groups,
    /*weight_tensor_groups=*/attrs_and_signature.weight_tensor_groups,
    /*output_tensor_groups=*/attrs_and_signature.output_tensor_groups,
  };
}

SymbolicCgOpAttrsAndTrainingSignatureWithShapes
  make_symbolic_cg_op_attrs_and_signature_with_shapes(
    ComputationGraphOpAttrs const &op_attrs,
    SymbolicLayerTrainingTensorGroupSignatureWithShapes const &signature) {
  return SymbolicCgOpAttrsAndTrainingSignatureWithShapes{
    /*op_attrs=*/op_attrs,
    /*input_tensor_groups=*/signature.input_tensor_groups,
    /*weight_tensor_groups=*/signature.weight_tensor_groups,
    /*output_tensor_groups=*/signature.output_tensor_groups,
  };
}

} // namespace FlexFlow
