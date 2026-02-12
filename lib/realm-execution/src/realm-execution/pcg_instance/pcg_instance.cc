#include "realm-execution/pcg_instance/pcg_instance.h"
#include "pcg/optimizer_attrs.h"
#include "realm-execution/dependency_set.h"
#include "realm-execution/distributed_device_state_initialization.h"
#include "realm-execution/instance_allocation.h"
#include "realm-execution/realm_context.h"
#include "realm-execution/tasks/impl/op_task.h"
#include "task-spec/dynamic_graph/dynamic_open_dataflow_graph.h"
#include "task-spec/dynamic_graph/dynamic_tensor_guid_t.dtg.h"
#include "task-spec/dynamic_graph/dynamic_value_attrs.dtg.h"
#include "task-spec/dynamic_graph/loss_insertion.h"
#include "task-spec/dynamic_graph/make_dynamic_open_dataflow_graph_from_mpcg.h"
#include "task-spec/dynamic_graph/pass_expansion.h"
#include "task-spec/dynamic_graph/shard_expansion.h"
#include "task-spec/dynamic_graph/update_insertion.h"
#include "utils/containers/transform.h"
#include "utils/containers/values.h"
#include "utils/graph/digraph/algorithms/get_topological_ordering.h"
#include "utils/optional.h"

namespace FlexFlow {

PCGInstance::PCGInstance(
    RealmContext &ctx,
    std::vector<DynamicNodeInvocation> const &execution_order,
    OptimizerAttrs const &optimizer_attrs,
    std::optional<Realm::RegionInstance> logit_grad_tensor)
    : ctx(ctx), execution_order(execution_order),
      optimizer_attrs(optimizer_attrs), logit_grad_tensor(logit_grad_tensor) {}

RealmContext &PCGInstance::get_realm_context() {
  return this->ctx;
}
std::vector<DynamicNodeInvocation> const &
    PCGInstance::get_execution_order() const {
  return this->execution_order;
}
OptimizerAttrs const &PCGInstance::get_optimizer_attrs() const {
  return this->optimizer_attrs;
}
void PCGInstance::update_optimizer_attrs_for_next_iter() {
  this->optimizer_attrs =
      get_optimizer_attrs_for_next_iter(this->optimizer_attrs);
}
std::optional<Realm::RegionInstance>
    PCGInstance::get_loss_tensor_instance() const {
  return this->logit_grad_tensor;
}

PCGInstance create_parallel_computation_graph_instance(
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

  std::optional<Realm::RegionInstance> logit_grad_tensor =
      transform(logit_grad_value, [&](DynamicValueAttrs const &lgv) {
        return backing.backing.at(lgv).first;
      });

  // FIXME: for now we're going to be lazy and block on everything rather than
  // do fine-grained dependencies on instances
  dg = perform_distributed_device_state_initialization(
      dg,
      ctx,
      profiling_settings,
      iteration_config,
      optimizer_attrs,
      ctx.get_outstanding_events());

  // Compute the topological ordering of the graph
  auto [kwarg_graph, node_map] =
      labelled_open_kwarg_dataflow_graph_from_dynamic_open_dataflow_graph(dg);
  std::vector<Node> node_topo_order = get_topological_ordering(kwarg_graph);
  std::vector<DynamicNodeInvocation> invocation_topo_order = transform(
      node_topo_order, [&](Node node) { return node_map.at_l(node); });

  return PCGInstance{
      ctx, invocation_topo_order, optimizer_attrs, logit_grad_tensor};

  // TODO list:
  //  * Realm allocator
  //  * external instances
  //  * task argument serializer
  //  * copies
}

static std::unordered_map<dynamic_layer_guid_t, Realm::Event>
    execute_distributed_dynamic_node_invocation_set(
        RealmContext &ctx,
        std::vector<DynamicNodeInvocation> const &invocations,
        OptimizerAttrs const &optimizer_attrs,
        ProfilingSettings const &profiling_settings,
        FFIterationConfig iteration_config) {
  // For simplicity we'll track a dependency on all outstanding operations up to
  // this point. This will create an effective barrier between phases.
  DependencySet dependency_set{ctx.get_outstanding_events()};
  return unordered_map_from_pairs(
      transform(invocations, [&](DynamicNodeInvocation const &invocation) {
        std::vector<Realm::Event> input_dependencies =
            transform(vector_of(values(invocation.inputs)),
                      [&](DynamicValueAttrs const &value) {
                        return dependency_set.get_dependency_for_reader(value);
                      });
        std::vector<Realm::Event> output_dependencies =
            transform(vector_of(values(invocation.outputs)),
                      [&](DynamicValueAttrs const &value) {
                        return dependency_set.get_dependency_for_writer(value);
                      });
        Realm::Event dependencies = Realm::Event::merge_events(
            Realm::Event::merge_events(input_dependencies),
            Realm::Event::merge_events(output_dependencies));
        Realm::Event result =
            spawn_op_task(ctx,
                          ctx.map_device_coord_to_processor(assert_unwrap(
                              invocation.node_attrs.device_coord)),
                          invocation,
                          profiling_settings,
                          iteration_config,
                          optimizer_attrs,
                          dependencies);
        for (DynamicValueAttrs const &value : values(invocation.inputs)) {
          dependency_set.add_reader(value, result);
        }
        for (DynamicValueAttrs const &value : values(invocation.outputs)) {
          dependency_set.add_writer(value, result);
        }
        return std::pair{invocation.node_attrs.layer_guid, result};
      }));
}

std::unordered_map<dynamic_layer_guid_t, Realm::Event>
    perform_all_passes_for_parallel_computation_graph_instance(
        PCGInstance &instance,
        ProfilingSettings const &profiling_settings,
        FFIterationConfig iteration_config) {
  std::vector<DynamicNodeInvocation> execution_order =
      instance.get_execution_order();
  std::unordered_map<dynamic_layer_guid_t, Realm::Event> result =
      execute_distributed_dynamic_node_invocation_set(
          /*ctx=*/instance.get_realm_context(),
          /*invocations=*/execution_order,
          /*optimizer_attrs=*/instance.get_optimizer_attrs(),
          /*profiling_settings=*/profiling_settings,
          /*iteration_config=*/iteration_config);
  instance.update_optimizer_attrs_for_next_iter();
  return result;
}

std::unordered_map<dynamic_layer_guid_t, Realm::Event>
    perform_forward_pass_for_parallel_computation_graph_instance(
        PCGInstance &instance,
        ProfilingSettings const &profiling_settings,
        FFIterationConfig iteration_config) {
  std::vector<DynamicNodeInvocation> execution_order =
      filter(instance.get_execution_order(),
             [](DynamicNodeInvocation const &invocation) {
               DynamicTaskType task_type =
                   assert_unwrap(invocation.node_attrs.task_type);
               return task_type == DynamicTaskType::FWD;
             });

  return execute_distributed_dynamic_node_invocation_set(
      /*ctx=*/instance.get_realm_context(),
      /*invocations=*/execution_order,
      /*optimizer_attrs=*/instance.get_optimizer_attrs(),
      /*profiling_settings=*/profiling_settings,
      /*iteration_config=*/iteration_config);
}

std::unordered_map<dynamic_layer_guid_t, Realm::Event>
    perform_backward_pass_for_parallel_computation_graph_instance(
        PCGInstance &instance,
        ProfilingSettings const &profiling_settings,
        FFIterationConfig iteration_config) {
  std::vector<DynamicNodeInvocation> execution_order =
      filter(instance.get_execution_order(),
             [](DynamicNodeInvocation const &invocation) {
               DynamicTaskType task_type =
                   assert_unwrap(invocation.node_attrs.task_type);
               return task_type == DynamicTaskType::BWD;
             });

  return execute_distributed_dynamic_node_invocation_set(
      /*ctx=*/instance.get_realm_context(),
      /*invocations=*/execution_order,
      /*optimizer_attrs=*/instance.get_optimizer_attrs(),
      /*profiling_settings=*/profiling_settings,
      /*iteration_config=*/iteration_config);
}

std::unordered_map<dynamic_layer_guid_t, Realm::Event>
    perform_update_pass_for_parallel_computation_graph_instance(
        PCGInstance &instance,
        ProfilingSettings const &profiling_settings,
        FFIterationConfig iteration_config) {
  std::vector<DynamicNodeInvocation> execution_order =
      filter(instance.get_execution_order(),
             [](DynamicNodeInvocation const &invocation) {
               DynamicTaskType task_type =
                   assert_unwrap(invocation.node_attrs.task_type);
               return task_type == DynamicTaskType::UPD;
             });

  std::unordered_map<dynamic_layer_guid_t, Realm::Event> result =
      execute_distributed_dynamic_node_invocation_set(
          /*ctx=*/instance.get_realm_context(),
          /*invocations=*/execution_order,
          /*optimizer_attrs=*/instance.get_optimizer_attrs(),
          /*profiling_settings=*/profiling_settings,
          /*iteration_config=*/iteration_config);
  instance.update_optimizer_attrs_for_next_iter();
  return result;
}

} // namespace FlexFlow
