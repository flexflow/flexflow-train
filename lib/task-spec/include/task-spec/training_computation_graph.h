#ifndef _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_TRAINING_COMPUTATION_GRAPH_H
#define _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_TRAINING_COMPUTATION_GRAPH_H

#include "pcg/optimizer_attrs.dtg.h"
#include "task-spec/forward_tensor_source.h"
#include "task-spec/gradient_tensor_source.h"
#include "task-spec/optimizer_tensor_source.h"
#include "task-spec/training_computation_graph.dtg.h"
#include "task-spec/training_layer_plus_context.dtg.h"
#include "task-spec/training_tensor_guid_t.dtg.h"

namespace FlexFlow {

TrainingComputationGraph generate_training_computation_graph(
    ComputationGraph const &computation_graph,
    OptimizerAttrs const &optimizer_attrs,
    ForwardTensorSource &forward_tensor_source,
    GradientTensorSource &gradient_tensor_source,
    OptimizerTensorSource &optimizer_tensor_source);

TrainingTensorGroup
    get_training_tensor_group_for_tensor_guid(TrainingComputationGraph const &,
                                              tensor_guid_t);
TrainingTensorGroupWithAttrs
    get_training_tensor_group_with_attrs_for_tensor_guid(
        TrainingComputationGraph const &, tensor_guid_t);

forward_tensor_guid_t
    get_forward_tensor_guid_for_tensor_guid(TrainingComputationGraph const &,
                                            tensor_guid_t);
gradient_tensor_guid_t
    get_gradient_tensor_guid_for_tensor_guid(TrainingComputationGraph const &,
                                             tensor_guid_t);
std::vector<optimizer_tensor_guid_t>
    get_optimizer_tensor_guids_for_tensor_guid(TrainingComputationGraph const &,
                                               tensor_guid_t);

tensor_guid_t
    get_tensor_guid_for_forward_tensor_guid(TrainingComputationGraph const &,
                                            forward_tensor_guid_t);
tensor_guid_t
    get_tensor_guid_for_gradient_tensor_guid(TrainingComputationGraph const &,
                                             gradient_tensor_guid_t);
tensor_guid_t
    get_tensor_guid_for_optimizer_tensor_guid(TrainingComputationGraph const &,
                                              optimizer_tensor_guid_t);

tensor_guid_t
    get_tensor_guid_for_training_tensor_guid(TrainingComputationGraph const &,
                                             training_tensor_guid_t);

std::unordered_set<training_tensor_guid_t>
    get_all_training_tensors_in_training_computation_graph(
        TrainingComputationGraph const &);

TrainingLayerPlusContext
    get_training_layer_plus_context(TrainingComputationGraph const &,
                                    layer_guid_t);

std::unordered_map<training_tensor_guid_t, TensorShape>
    get_all_training_tensor_shapes(TrainingComputationGraph const &);

} // namespace FlexFlow

#endif
