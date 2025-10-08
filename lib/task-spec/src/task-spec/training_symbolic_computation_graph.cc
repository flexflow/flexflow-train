#include "task-spec/training_symbolic_computation_graph.h"
#include "pcg/computation_graph.h"
#include "pcg/parallel_computation_graph/parallel_layer_attrs.h"
#include "task-spec/lower_op_task_invocation_to_runtime_task_invocation.h"
#include "task-spec/symbolic_loss_tensor_source.h"
#include "task-spec/symbolic_training_tensor_group.h"
#include "task-spec/task_signature_impl.h"
#include "utils/bidict/generate_bidict.h"
#include "utils/containers/contains.h"
#include "utils/containers/filter_values.h"
#include "utils/containers/flatmap.h"
#include "utils/containers/generate_map.h"
#include "utils/containers/get_only.h"
#include "utils/containers/keys.h"
#include "utils/containers/set_of.h"
#include "utils/containers/transform.h"
#include "utils/overload.h"
#include "task-spec/loss_functions.h"
#include "task-spec/optimizer.h"

namespace FlexFlow {

TensorShape get_symbolic_tensor_shape(TrainingSymbolicComputationGraph const &g,
                                      symbolic_tensor_guid_t t) {
  return g.symbolic_computation_graph.at(t.raw_graph_output);
}

PCGOperatorAttrs get_op_attrs_for_symbolic_layer_guid(TrainingSymbolicComputationGraph const &,
                                                      symbolic_layer_guid_t) {
  NOT_IMPLEMENTED();
}

TrainingLayerSymbolicTensorGroupSignatureWithShapes
  get_signature_with_shapes_for_symbolic_layer_guid(TrainingSymbolicComputationGraph const &,
                                                    symbolic_layer_guid_t) {
  NOT_IMPLEMENTED();
}

symbolic_forward_tensor_guid_t get_forward_symbolic_tensor_guid_for_symbolic_tensor_guid(
    TrainingSymbolicComputationGraph const &g, symbolic_tensor_guid_t t) {
  return g.symbolic_training_tensor_group_for_tensor.at(t).forward_tensor;
}

symbolic_gradient_tensor_guid_t get_gradient_symbolic_tensor_guid_for_symbolic_tensor_guid(
    TrainingSymbolicComputationGraph const &, tensor_guid_t) {
  NOT_IMPLEMENTED();
}

std::vector<symbolic_optimizer_tensor_guid_t> get_optimizer_symbolic_tensor_guids_for_symbolic_tensor_guid(
    TrainingSymbolicComputationGraph const &, tensor_guid_t) {
  NOT_IMPLEMENTED();
}

symbolic_tensor_guid_t get_symbolic_tensor_guid_for_forward_symbolic_tensor_guid(
    TrainingSymbolicComputationGraph const &, symbolic_forward_tensor_guid_t) {
  NOT_IMPLEMENTED();
}

symbolic_tensor_guid_t get_symbolic_tensor_guid_for_gradient_symbolic_tensor_guid(
    TrainingSymbolicComputationGraph const &, symbolic_gradient_tensor_guid_t) {
  NOT_IMPLEMENTED();
}

symbolic_tensor_guid_t get_symbolic_tensor_guid_for_optimizer_symbolic_tensor_guid(
    TrainingSymbolicComputationGraph const &, symbolic_optimizer_tensor_guid_t) {
  NOT_IMPLEMENTED();
}

symbolic_tensor_guid_t get_symbolic_tensor_guid_for_training_symbolic_tensor_guid(
    TrainingSymbolicComputationGraph const &, symbolic_training_tensor_guid_t) {
  NOT_IMPLEMENTED();
}

std::unordered_set<symbolic_training_tensor_guid_t>
    get_all_symbolic_training_tensors_in_training_computation_graph(
        TrainingSymbolicComputationGraph const &) {
  NOT_IMPLEMENTED();
}

SymbolicTrainingLayerAttrsPlusContext
    get_symbolic_training_layer_attrs_plus_context(TrainingSymbolicComputationGraph const &,
                                    symbolic_layer_guid_t) {
  NOT_IMPLEMENTED();
}

std::unordered_map<symbolic_training_tensor_guid_t, TensorShape>
    get_all_symbolic_training_tensor_shapes(TrainingSymbolicComputationGraph const &) {
  NOT_IMPLEMENTED();
}

static ComputationGraphOpAttrs get_cg_op_attrs_for_symbolic_layer_guid(TrainingSymbolicComputationGraph const &g,
                                               symbolic_layer_guid_t l) {
  PCGOperatorAttrs op_attrs = get_op_attrs_for_symbolic_layer_guid(g, l);
  std::optional<ComputationGraphOpAttrs>
    cg_op_attrs = compgraph_op_attrs_from_pcg_op_attrs(op_attrs);

  ASSERT(cg_op_attrs.has_value());
  
  return cg_op_attrs.value();
}

std::optional<RuntimeTaskInvocation>
  get_init_task_invocation_for_layer(TrainingSymbolicComputationGraph const &g,
                                     symbolic_layer_guid_t l) {
  ComputationGraphOpAttrs cg_op_attrs = get_cg_op_attrs_for_symbolic_layer_guid(g, l);

  OpTaskInvocation op_task_invocation = ({
    std::optional<OpTaskInvocation> maybe_invocation = get_init_op_task_invocation(cg_op_attrs);
    if (!maybe_invocation.has_value()) {
      return std::nullopt;
    }
    maybe_invocation.value();
  });

  TrainingLayerSymbolicTensorGroupSignatureWithShapes layer_signature 
    = get_signature_with_shapes_for_symbolic_layer_guid(g, l);

  return lower_op_task_invocation_to_runtime_task_invocation(
    /*op_task_invocation=*/op_task_invocation,
    /*symbolic_layer_guid=*/l,
    /*layer_signature=*/layer_signature);
}

std::optional<RuntimeTaskInvocation>
  get_forward_runtime_task_invocation_for_layer(TrainingSymbolicComputationGraph const &g,
                                        symbolic_layer_guid_t l) {
  ComputationGraphOpAttrs cg_op_attrs = get_cg_op_attrs_for_symbolic_layer_guid(g, l);

  OpTaskInvocation op_task_invocation = ({
    std::optional<OpTaskInvocation> maybe_invocation = get_forward_op_task_invocation(cg_op_attrs);
    if (!maybe_invocation.has_value()) {
      return std::nullopt;
    }
    maybe_invocation.value();
  });

  TrainingLayerSymbolicTensorGroupSignatureWithShapes layer_signature 
    = get_signature_with_shapes_for_symbolic_layer_guid(g, l);

  return lower_op_task_invocation_to_runtime_task_invocation(
    /*op_task_invocation=*/op_task_invocation,
    /*symbolic_layer_guid=*/l,
    /*layer_signature=*/layer_signature);
}

std::optional<RuntimeTaskInvocation>
  get_backward_task_invocation_for_layer(TrainingSymbolicComputationGraph const &g,
                                         symbolic_layer_guid_t l) {
  ComputationGraphOpAttrs cg_op_attrs = get_cg_op_attrs_for_symbolic_layer_guid(g, l);

  OpTaskInvocation op_task_invocation = ({
    std::optional<OpTaskInvocation> maybe_invocation = get_backward_op_task_invocation(cg_op_attrs);
    if (!maybe_invocation.has_value()) {
      return std::nullopt;
    }
    maybe_invocation.value();
  });

  TrainingLayerSymbolicTensorGroupSignatureWithShapes layer_signature 
    = get_signature_with_shapes_for_symbolic_layer_guid(g, l);

  return lower_op_task_invocation_to_runtime_task_invocation(
    /*op_task_invocation=*/op_task_invocation,
    /*symbolic_layer_guid=*/l,
    /*layer_signature=*/layer_signature);
}

RuntimeTaskInvocation
  get_compute_loss_runtime_task_invocation(TrainingSymbolicComputationGraph const &g) {

  symbolic_tensor_guid_t logit_tensor = g.logit_tensor;
  symbolic_loss_tensor_guid_t label_tensor = g.label_tensor;

  RuntimeTaskInvocation loss_invocation = loss_attrs_backward(
      g.loss_attrs,
      get_forward_symbolic_tensor_guid_for_symbolic_tensor_guid(g, logit_tensor),
      get_gradient_symbolic_tensor_guid_for_symbolic_tensor_guid(g, logit_tensor),
      label_tensor);

  return loss_invocation;
} 

std::optional<RuntimeTaskInvocation>
  get_update_runtime_task_invocation_for_layer(TrainingSymbolicComputationGraph const &g,
                                               symbolic_layer_guid_t l) {
  SymbolicTrainingLayerAttrsPlusContext training_layer = get_symbolic_training_layer_attrs_plus_context(
      g, l);

  if (training_layer.layer_attrs.op_attrs.has<WeightAttrs>()) {
    SymbolicTrainingTensorGroup weight_tensor_group =
        get_only(training_layer.output_tensor_groups);

    RuntimeTaskInvocation invocation =
        optimizer_attrs_get_update_invocation(
          g.optimizer_attrs,
          /*weight=*/weight_tensor_group.forward_tensor,
          /*weight_grad=*/weight_tensor_group.gradient_tensor,
          /*grad_buffer_tensors=*/weight_tensor_group.optimizer_tensors);

    return invocation;
  } else {
    return std::nullopt;
  }
}

} // namespace FlexFlow
