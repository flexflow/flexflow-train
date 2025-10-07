#ifndef _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_TRAINING_SYMBOLIC_COMPUTATION_GRAPH_H
#define _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_TRAINING_SYMBOLIC_COMPUTATION_GRAPH_H

#include "task-spec/symbolic_forward_tensor_source.h"
#include "task-spec/symbolic_gradient_tensor_source.h"
#include "task-spec/symbolic_loss_tensor_source.h"
#include "task-spec/symbolic_optimizer_tensor_source.h"
#include "task-spec/training_symbolic_computation_graph.dtg.h"
#include "task-spec/symbolic_training_layer_plus_context.dtg.h"
#include "task-spec/symbolic_training_tensor_guid_t.dtg.h"
#include "task-spec/symbolic_training_tensor_group_with_attrs.dtg.h"
#include "task-spec/training_symbolic_computation_graph_from_cg_conversion.dtg.h"
#include "task-spec/training_symbolic_computation_graph_from_pcg_conversion.dtg.h"
#include "task-spec/symbolic_layer_guid_t.dtg.h"

namespace FlexFlow {

TensorShape get_symbolic_tensor_shape(TrainingSymbolicComputationGraph const &,
                                      symbolic_tensor_guid_t);

symbolic_forward_tensor_guid_t get_forward_tensor_guid_for_tensor_guid(
    TrainingSymbolicComputationGraph const &, symbolic_tensor_guid_t);

symbolic_gradient_tensor_guid_t get_gradient_tensor_guid_for_tensor_guid(
    TrainingSymbolicComputationGraph const &, tensor_guid_t);

std::vector<symbolic_optimizer_tensor_guid_t> get_optimizer_tensor_guids_for_tensor_guid(
    TrainingSymbolicComputationGraph const &, tensor_guid_t);

symbolic_tensor_guid_t get_symbolic_tensor_guid_for_forward_tensor_guid(
    TrainingSymbolicComputationGraph const &, symbolic_forward_tensor_guid_t);

symbolic_tensor_guid_t get_symbolic_tensor_guid_for_gradient_tensor_guid(
    TrainingSymbolicComputationGraph const &, symbolic_gradient_tensor_guid_t);

symbolic_tensor_guid_t get_symbolic_tensor_guid_for_optimizer_tensor_guid(
    TrainingSymbolicComputationGraph const &, symbolic_optimizer_tensor_guid_t);

symbolic_tensor_guid_t get_symbolic_tensor_guid_for_training_tensor_guid(
    TrainingSymbolicComputationGraph const &, symbolic_training_tensor_guid_t);

std::unordered_set<symbolic_training_tensor_guid_t>
    get_all_training_tensors_in_training_computation_graph(
        TrainingSymbolicComputationGraph const &);

SymbolicTrainingLayerPlusContext
    get_training_layer_plus_context(TrainingSymbolicComputationGraph const &,
                                    symbolic_layer_guid_t);

std::unordered_map<symbolic_training_tensor_guid_t, TensorShape>
    get_all_training_tensor_shapes(TrainingSymbolicComputationGraph const &);

} // namespace FlexFlow

#endif

