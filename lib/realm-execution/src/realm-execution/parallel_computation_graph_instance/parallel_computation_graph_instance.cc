#include "realm-execution/parallel_computation_graph_instance/parallel_computation_graph_instance.h"
#include "pcg/optimizer_attrs.h"
#include "realm-execution/distributed_device_state_initialization.h"
#include "realm-execution/instance_allocation.h"
#include "task-spec/dynamic_graph/dynamic_open_dataflow_graph.h"
#include "task-spec/dynamic_graph/dynamic_tensor_guid_t.dtg.h"
#include "task-spec/dynamic_graph/loss_insertion.h"
#include "task-spec/dynamic_graph/make_dynamic_open_dataflow_graph_from_mpcg.h"
#include "task-spec/dynamic_graph/pass_expansion.h"
#include "task-spec/dynamic_graph/shard_expansion.h"
#include "task-spec/dynamic_graph/update_insertion.h"
#include "utils/exception.h"
#include "utils/optional.h"

namespace FlexFlow {

ParallelComputationGraphInstance::ParallelComputationGraphInstance(
    RealmContext &realm,
    DynamicOpenDataflowGraph dataflow_graph,
    std::vector<DynamicNodeInvocation> const &topological_ordering,
    OptimizerAttrs const &optimizer_attrs,
    std::optional<LossAttrs> const &loss_attrs,
    std::optional<GenericTensorAccessorW> logit_grad_tensor)
    : realm(realm), dataflow_graph(dataflow_graph),
      topological_ordering(topological_ordering),
      optimizer_attrs(optimizer_attrs), loss_attrs(loss_attrs),
      logit_grad_tensor(logit_grad_tensor) {}

DynamicOpenDataflowGraph const &
    ParallelComputationGraphInstance::get_dynamic_dataflow_graph() const {
  return this->dataflow_graph;
}
Allocator &ParallelComputationGraphInstance::get_allocator() const {
  return this->realm.get_current_device_allocator();
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
    RealmContext &ctx,
    MappedParallelComputationGraph const &mpcg,
    OptimizerAttrs const &optimizer_attrs,
    std::optional<LossAttrs> const &loss_attrs,
    std::optional<GenericTensorAccessorR> label_tensor,
    std::optional<parallel_tensor_guid_t> logit_tensor,
    std::unordered_map<DynamicValueAttrs, DynamicTensorAccessor> const
        &input_tensors,
    ProfilingSettings const &profiling_settings,
    FFIterationConfig const &iteration_config) {

  DynamicOpenDataflowGraph dg =
      make_dynamic_open_dataflow_graph_from_mpcg(mpcg);
  dg = perform_pass_expansion(dg);

  std::unordered_map<DynamicValueAttrs, DynamicTensorAccessor> inputs =
      input_tensors;
  std::optional<DynamicValueAttrs> logit_grad_value;
  if (loss_attrs) {
    auto [dg2, label_v, logit_grad_v] = perform_loss_insertion(
        dg,
        assert_unwrap(loss_attrs),
        dynamic_tensor_guid_t{assert_unwrap(logit_tensor)});
    dg = dg2;
    logit_grad_value = logit_grad_v;
    inputs.insert(std::pair{label_v, assert_unwrap(label_tensor)});
  }

  dg = perform_update_insertion(dg, optimizer_attrs);
  dg = perform_shard_expansion(dg);
  TensorInstanceBacking backing = perform_instance_allocation(dg, inputs, ctx);

  // FIXME: for now we're going to be lazy and block on everything rather than
  // do fine-grained dependencies
  ctx.get_outstanding_events().wait();

  std::optional<Realm::RegionInstance> logit_grad_tensor =
      transform(logit_grad_value, [&](DynamicValueAttrs const &lgv) {
        return backing.backing.at(lgv).first;
      });

  dg = perform_distributed_device_state_initialization(
      dg, ctx, profiling_settings, iteration_config, optimizer_attrs);
  NOT_IMPLEMENTED();

  // TODO list:
  //  * per-device state initialization (RPC mechanism?)
  //  * Realm allocator
  //  * task body
  //  * external instances
}

} // namespace FlexFlow
