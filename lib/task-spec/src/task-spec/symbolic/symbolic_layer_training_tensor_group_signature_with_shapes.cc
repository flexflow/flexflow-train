#include "task-spec/symbolic/symbolic_layer_training_tensor_group_signature_with_shapes.h"
#include "task-spec/symbolic/symbolic_training_tensor_group_with_shape.h"
#include "utils/containers/transform.h"

namespace FlexFlow {


SymbolicLayerTrainingTensorGroupSignature
  drop_shapes_from_signature(SymbolicLayerTrainingTensorGroupSignatureWithShapes const &s) {

  auto drop_shapes = [](std::vector<SymbolicTrainingTensorGroupWithShape> const &groups) {
    return transform(groups, 
                     [](SymbolicTrainingTensorGroupWithShape const &g) {
                       return drop_shape_from_group(g);
                     });
  };

  return SymbolicLayerTrainingTensorGroupSignature{
    /*input_tensor_groups=*/drop_shapes(s.input_tensor_groups),
    /*weight_tensor_groups=*/drop_shapes(s.weight_tensor_groups),
    /*output_tensor_groups=*/drop_shapes(s.output_tensor_groups),
  };
}

SymbolicLayerTensorShapeSignature
  get_shape_signature(SymbolicLayerTrainingTensorGroupSignatureWithShapes const &s) {

  auto get_shapes = [](std::vector<SymbolicTrainingTensorGroupWithShape> const &groups) {
    return transform(groups, 
                     [](SymbolicTrainingTensorGroupWithShape const &g) {
                       return g.tensor_shape;
                     });
  };

  return SymbolicLayerTensorShapeSignature{
    /*input_shapes=*/get_shapes(s.input_tensor_groups),
    /*weight_shapes=*/get_shapes(s.weight_tensor_groups),
    /*output_shapes=*/get_shapes(s.output_tensor_groups),
  };
}


} // namespace FlexFlow
