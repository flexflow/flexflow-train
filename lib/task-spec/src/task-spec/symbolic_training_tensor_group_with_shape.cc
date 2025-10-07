#include "task-spec/symbolic_training_tensor_group_with_shape.h"

namespace FlexFlow {

SymbolicTrainingTensorGroup
  drop_shape_from_group(SymbolicTrainingTensorGroupWithShape const &g) {

  return g.training_tensor_group;
}


} // namespace FlexFlow
