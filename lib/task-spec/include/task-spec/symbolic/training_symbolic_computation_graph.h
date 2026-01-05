#ifndef _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_SYMBOLIC_TRAINING_SYMBOLIC_COMPUTATION_GRAPH_H
#define _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_SYMBOLIC_TRAINING_SYMBOLIC_COMPUTATION_GRAPH_H

#include "task-spec/ops/op_task_type.dtg.h"
#include "task-spec/symbolic/symbolic_cg_op_attrs_and_training_signature_with_shapes.dtg.h"
#include "task-spec/symbolic/symbolic_forward_tensor_source.h"
#include "task-spec/symbolic/symbolic_gradient_tensor_source.h"
#include "task-spec/symbolic/symbolic_loss_tensor_source.h"
#include "task-spec/symbolic/symbolic_optimizer_tensor_source.h"
#include "task-spec/symbolic/symbolic_layer_training_tensor_group_signature_with_shapes.dtg.h"
#include "task-spec/symbolic/training_symbolic_computation_graph.dtg.h"
#include "task-spec/symbolic/symbolic_training_layer_attrs_plus_context.dtg.h"
#include "task-spec/symbolic/symbolic_training_tensor_guid_t.dtg.h"
#include "task-spec/symbolic/symbolic_training_tensor_group_with_attrs.dtg.h"
#include "task-spec/symbolic/training_symbolic_computation_graph_from_cg_conversion.dtg.h"
#include "task-spec/symbolic/training_symbolic_computation_graph_from_pcg_conversion.dtg.h"
#include "task-spec/symbolic/symbolic_layer_guid_t.dtg.h"
#include "task-spec/symbolic/symbolic_cg_op_attrs_and_training_signature_with_shapes.dtg.h"

namespace FlexFlow {

TensorShape get_symbolic_tensor_shape(TrainingSymbolicComputationGraph const &,
                                      symbolic_tensor_guid_t);

PCGOperatorAttrs get_op_attrs_for_symbolic_layer_guid(TrainingSymbolicComputationGraph const &,
                                                      symbolic_layer_guid_t);

SymbolicLayerTrainingTensorGroupSignatureWithShapes
  get_signature_with_shapes_for_symbolic_layer_guid(TrainingSymbolicComputationGraph const &,
                                                    symbolic_layer_guid_t);

symbolic_forward_tensor_guid_t get_forward_symbolic_tensor_guid_for_symbolic_tensor_guid(
    TrainingSymbolicComputationGraph const &, symbolic_tensor_guid_t);

symbolic_gradient_tensor_guid_t get_gradient_symbolic_tensor_guid_for_symbolic_tensor_guid(
    TrainingSymbolicComputationGraph const &, symbolic_tensor_guid_t);

std::vector<symbolic_optimizer_tensor_guid_t> get_optimizer_tensor_guids_for_symbolic_tensor_guid(
    TrainingSymbolicComputationGraph const &, symbolic_tensor_guid_t);

symbolic_tensor_guid_t get_symbolic_tensor_guid_for_forward_symbolic_tensor_guid(
    TrainingSymbolicComputationGraph const &, symbolic_forward_tensor_guid_t);

symbolic_tensor_guid_t get_symbolic_tensor_guid_for_gradient_symbolic_tensor_guid(
    TrainingSymbolicComputationGraph const &, symbolic_gradient_tensor_guid_t);

symbolic_tensor_guid_t get_symbolic_tensor_guid_for_optimizer_symbolic_tensor_guid(
    TrainingSymbolicComputationGraph const &, symbolic_optimizer_tensor_guid_t);

symbolic_tensor_guid_t get_symbolic_tensor_guid_for_training_symbolic_tensor_guid(
    TrainingSymbolicComputationGraph const &, symbolic_training_tensor_guid_t);

std::unordered_set<symbolic_training_tensor_guid_t>
    get_all_symbolic_training_tensors_in_training_computation_graph(
        TrainingSymbolicComputationGraph const &);

std::vector<symbolic_layer_guid_t>
  symbolic_cg_topological_ordering(TrainingSymbolicComputationGraph const &);

SymbolicTrainingLayerAttrsPlusContext
    get_symbolic_training_layer_attrs_plus_context(TrainingSymbolicComputationGraph const &,
                                    symbolic_layer_guid_t);

std::unordered_map<symbolic_training_tensor_guid_t, TensorShape>
    get_all_symbolic_training_tensor_shapes(TrainingSymbolicComputationGraph const &);

SymbolicCgOpAttrsAndTrainingSignatureWithShapes 
  get_attrs_and_signature_for_layer(TrainingSymbolicComputationGraph const &,
                                    symbolic_layer_guid_t);

} // namespace FlexFlow

#endif

