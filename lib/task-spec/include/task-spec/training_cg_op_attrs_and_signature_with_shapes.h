#ifndef _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_TRAINING_CG_OP_ATTRS_AND_SIGNATURE_WITH_SHAPES_H
#define _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_TRAINING_CG_OP_ATTRS_AND_SIGNATURE_WITH_SHAPES_H

#include "task-spec/training_cg_op_attrs_and_signature_with_shapes.dtg.h"
#include "task-spec/training_layer_symbolic_tensor_group_signature_with_shapes.dtg.h"

namespace FlexFlow {

TrainingLayerSymbolicTensorGroupSignatureWithShapes
  get_signature_with_shapes(TrainingCgOpAttrsAndSignatureWithShapes const &);

TrainingCgOpAttrsAndSignatureWithShapes
  make_training_cg_op_attrs_and_signature(
    ComputationGraphOpAttrs const &,
    TrainingLayerSymbolicTensorGroupSignatureWithShapes const &);

} // namespace FlexFlow

#endif
