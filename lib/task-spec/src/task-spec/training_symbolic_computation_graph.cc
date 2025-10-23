#include "task-spec/training_symbolic_computation_graph.h"
#include "pcg/computation_graph.h"
#include "pcg/parallel_computation_graph/parallel_layer_attrs.h"
#include "task-spec/lower_op_task_invocation_to_runtime_task_invocation.h"
#include "task-spec/symbolic_cg_op_attrs_and_training_signature_with_shapes.h"
#include "task-spec/symbolic_loss_tensor_source.h"
#include "task-spec/symbolic_training_tensor_group.h"
#include "task-spec/task_signature_impl.h"
#include "task-spec/symbolic_cg_op_attrs_and_training_signature_with_shapes.h"
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

SymbolicLayerTrainingTensorGroupSignatureWithShapes
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

SymbolicCgOpAttrsAndTrainingSignatureWithShapes 
  get_attrs_and_signature_for_layer(TrainingSymbolicComputationGraph const &g,
                                    symbolic_layer_guid_t l) {

  ComputationGraphOpAttrs cg_op_attrs = get_cg_op_attrs_for_symbolic_layer_guid(g, l);

  SymbolicLayerTrainingTensorGroupSignatureWithShapes layer_signature 
    = get_signature_with_shapes_for_symbolic_layer_guid(g, l);

  return make_symbolic_cg_op_attrs_and_signature_with_shapes(cg_op_attrs, layer_signature);
}

} // namespace FlexFlow
