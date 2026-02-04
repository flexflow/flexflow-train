#include "realm-execution/parallel_computation_graph_instance/parallel_computation_graph_instance.h"
#include "pcg/optimizer_attrs.h"
#include "utils/exception.h"

namespace FlexFlow {

ParallelComputationGraphInstance::ParallelComputationGraphInstance(
    DynamicOpenDataflowGraph dataflow_graph,
    Allocator &allocator,
    std::vector<DynamicNodeInvocation> const &topological_ordering,
    OptimizerAttrs const &optimizer_attrs,
    std::optional<LossAttrs> const &loss_attrs,
    std::optional<GenericTensorAccessorW> logit_grad_tensor)
    : dataflow_graph(dataflow_graph), allocator(allocator),
      topological_ordering(topological_ordering),
      optimizer_attrs(optimizer_attrs), loss_attrs(loss_attrs),
      logit_grad_tensor(logit_grad_tensor) {}

DynamicOpenDataflowGraph const &
    ParallelComputationGraphInstance::get_dynamic_dataflow_graph() const {
  return this->dataflow_graph;
}
Allocator &ParallelComputationGraphInstance::get_allocator() const {
  return this->allocator;
}
std::vector<DynamicNodeInvocation> const &
    ParallelComputationGraphInstance::get_topological_ordering() const {
  return this->topological_ordering;
}
OptimizerAttrs const &
    ParallelComputationGraphInstance::get_optimizer_attrs() const {
  return this->optimizer_attrs;
}
void ParallelComputationGraphInstance::update_optimizer_attrs_for_next_iter() {
  this->optimizer_attrs =
      get_optimizer_attrs_for_next_iter(this->optimizer_attrs);
}
std::optional<LossAttrs> const &
    ParallelComputationGraphInstance::get_loss_attrs() const {
  return this->loss_attrs;
}
std::optional<GenericTensorAccessorR>
    ParallelComputationGraphInstance::get_loss_tensor_accessor() const {
  return this->logit_grad_tensor;
}

ParallelComputationGraphInstance create_parallel_computation_graph_instance(
    ParallelComputationGraph const &pcg,
    OptimizerAttrs const &optimizer_attrs,
    std::optional<LossAttrs> const &loss_attrs,
    std::optional<GenericTensorAccessorR> label_tensor,
    std::optional<dynamic_tensor_guid_t> logit_tensor,
    std::unordered_map<DynamicValueAttrs, DynamicTensorAccessor> const
        &input_tensors,
    Allocator &allocator,
    ProfilingSettings const &profiling_settings,
    device_handle_t const &device_handle,
    FFIterationConfig const &iteration_config,
    device_id_t device_idx) {
  NOT_IMPLEMENTED();
}

} // namespace FlexFlow
