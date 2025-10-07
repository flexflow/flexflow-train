#include "task-spec/training_symbolic_computation_graph.h"
#include "pcg/computation_graph.h"
#include "pcg/parallel_computation_graph/parallel_layer_attrs.h"
#include "task-spec/symbolic_loss_tensor_source.h"
#include "task-spec/symbolic_training_tensor_group.h"
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

namespace FlexFlow {

TensorShape get_symbolic_tensor_shape(TrainingSymbolicComputationGraph const &g,
                                      symbolic_tensor_guid_t t) {
  return g.symbolic_computation_graph.at(t.raw_graph_output);
}

symbolic_forward_tensor_guid_t get_forward_tensor_guid_for_tensor_guid(
    TrainingSymbolicComputationGraph const &g, symbolic_tensor_guid_t t) {
  return g.symbolic_training_tensor_group_for_tensor.at(t).forward_tensor;
}

symbolic_gradient_tensor_guid_t get_gradient_tensor_guid_for_tensor_guid(
    TrainingSymbolicComputationGraph const &, tensor_guid_t) {
  NOT_IMPLEMENTED();
}

std::vector<symbolic_optimizer_tensor_guid_t> get_optimizer_tensor_guids_for_tensor_guid(
    TrainingSymbolicComputationGraph const &, tensor_guid_t) {
  NOT_IMPLEMENTED();
}

symbolic_tensor_guid_t get_symbolic_tensor_guid_for_forward_tensor_guid(
    TrainingSymbolicComputationGraph const &, symbolic_forward_tensor_guid_t) {
  NOT_IMPLEMENTED();
}

symbolic_tensor_guid_t get_symbolic_tensor_guid_for_gradient_tensor_guid(
    TrainingSymbolicComputationGraph const &, symbolic_gradient_tensor_guid_t) {
  NOT_IMPLEMENTED();
}

symbolic_tensor_guid_t get_symbolic_tensor_guid_for_optimizer_tensor_guid(
    TrainingSymbolicComputationGraph const &, symbolic_optimizer_tensor_guid_t) {
  NOT_IMPLEMENTED();
}

symbolic_tensor_guid_t get_symbolic_tensor_guid_for_training_tensor_guid(
    TrainingSymbolicComputationGraph const &, symbolic_training_tensor_guid_t) {
  NOT_IMPLEMENTED();
}

std::unordered_set<symbolic_training_tensor_guid_t>
    get_all_training_tensors_in_training_computation_graph(
        TrainingSymbolicComputationGraph const &) {
  NOT_IMPLEMENTED();
}

SymbolicTrainingLayerPlusContext
    get_training_layer_plus_context(TrainingSymbolicComputationGraph const &,
                                    symbolic_layer_guid_t) {
  NOT_IMPLEMENTED();
}

std::unordered_map<symbolic_training_tensor_guid_t, TensorShape>
    get_all_training_tensor_shapes(TrainingSymbolicComputationGraph const &) {
  NOT_IMPLEMENTED();
}

} // namespace FlexFlow
