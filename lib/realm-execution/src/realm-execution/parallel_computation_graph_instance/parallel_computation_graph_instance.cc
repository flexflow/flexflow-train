#include "realm-execution/parallel_computation_graph_instance/parallel_computation_graph_instance.h"
#include "local-execution/device_state_initialization.h"
#include "local-execution/tensor_allocation.h"
#include "pcg/optimizer_attrs.h"
#include "task-spec/dynamic_graph/dynamic_open_dataflow_graph.h"
#include "task-spec/dynamic_graph/loss_insertion.h"
#include "task-spec/dynamic_graph/make_dynamic_open_dataflow_graph_from_pcg.h"
#include "task-spec/dynamic_graph/pass_expansion.h"
#include "task-spec/dynamic_graph/update_insertion.h"
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

static GenericTensorAccessorW
    get_loss_tensor_accessor(DynamicOpenDataflowGraph const &dg,
                             DynamicValueAttrs const &value) {
  return find_output_tensor(dg, value.tensor_guid, value.role)
      .value()
      .second.accessor.value()
      .get<GenericTensorAccessorW>();
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

  DynamicOpenDataflowGraph dg = make_dynamic_open_dataflow_graph_from_pcg(pcg);
  dg = perform_pass_expansion(dg);

  std::unordered_map<DynamicValueAttrs, DynamicTensorAccessor> inputs =
      input_tensors;
  std::optional<DynamicValueAttrs> logit_grad_value;
  if (loss_attrs) {
    auto [dg2, label_v, logit_grad_v] = perform_loss_insertion(
        dg, assert_unwrap(loss_attrs), assert_unwrap(logit_tensor));
    dg = dg2;
    logit_grad_value = logit_grad_v;
    inputs.insert(std::pair{label_v, assert_unwrap(label_tensor)});
  }

  dg = perform_update_insertion(dg, optimizer_attrs);
  dg = perform_tensor_allocation(dg, inputs, allocator);

  std::optional<GenericTensorAccessorW> logit_grad_tensor =
      transform(logit_grad_value, [&](DynamicValueAttrs const &lgv) {
        return get_loss_tensor_accessor(dg, lgv);
      });

  dg = perform_device_state_initialization(dg,
                                           allocator,
                                           profiling_settings,
                                           device_handle,
                                           iteration_config,
                                           optimizer_attrs,
                                           device_idx);
  NOT_IMPLEMENTED();
}

} // namespace FlexFlow
