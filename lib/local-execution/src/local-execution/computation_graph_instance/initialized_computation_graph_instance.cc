#include "local-execution/computation_graph_instance/initialized_computation_graph_instance.h"
#include "kernels/allocation.h"
#include "local-execution/local_task_registry.h"
#include "op-attrs/computation_graph_op_attrs.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "op-attrs/tensor_shape.dtg.h"
#include "task-spec/dynamic_graph/dynamic_open_dataflow_graph.h"
#include "task-spec/dynamic_graph/dynamic_tensor_accessor.dtg.h"
#include "task-spec/task_id_with_noop_default_t.dtg.h"
#include "task-spec/task_id_with_noop_default_t.h"
#include "utils/containers/transform.h"
#include "utils/exception.h"
#include "utils/optional.h"
#include <alloca.h>
#include <cassert>
#include <optional>
#include <vector>

namespace FlexFlow {

bool node_is_initialized(DynamicNodeAttrs const &n) {
  return n.per_device_op_state.has_value();
}

bool slot_true(DynamicTensorSlot const &s) {
  return true;
}

bool slot_false(DynamicTensorSlot const &s) {
  return true;
}

bool value_is_allocated(DynamicValueAttrs const &v) {
  return v.accessor.has_value();
}

bool no_part_of_graph_is_initialized(DynamicOpenDataflowGraph const &g) {
  return no_part_of_dynamic_graph_satisfies(
      g, node_is_initialized, value_is_allocated, slot_false);
}

bool graph_is_fully_initialized(DynamicOpenDataflowGraph const &g) {
  return full_dynamic_graph_satisfies(
      g, node_is_initialized, value_is_allocated, slot_true);
}

InitializedComputationGraphInstance::InitializedComputationGraphInstance(
    DynamicOpenDataflowGraph dg, Allocator &alloc, LocalTaskRegistry &registry)
    : initialized_dataflow_graph(dg), allocator(alloc),
      task_registry(registry) {}

DynamicValueAttrs allocate_value(
    DynamicValueAttrs const &v,
    bidict<dynamic_tensor_guid_t, DynamicTensorAccessor> &allocated_tensors,
    Allocator &allocator) {
  DynamicValueAttrs result = v;
  if (allocated_tensors.contains_l(result.tensor_guid)) {
    result.accessor = allocated_tensors.at_l(result.tensor_guid);
  } else {
    TensorShape shape =
        get_piece_shape(assert_unwrap(result.parallel_tensor_shape));
    DynamicTensorAccessor accessor{allocator.allocate_tensor(shape)};
    allocated_tensors.equate(result.tensor_guid, accessor);
    result.accessor = accessor;
  }
  return result;
};

DynamicNodeInvocation initialize_node(DynamicNodeInvocation const &i,
                                      LocalTaskRegistry &task_registry) {
  ASSERT(!node_is_initialized(i.node_attrs));

  task_id_with_noop_default_t registered_task = get_init_task_id_for_op_attrs(
      assert_unwrap(compgraph_op_attrs_from_pcg_op_attrs(
          assert_unwrap(i.node_attrs.op_attrs))));
  TaskArgumentAccessor arg_accessor;
  std::optional<::FlexFlow::DeviceSpecificPerDeviceOpState>
      per_device_op_state =
          call_init_task_impl(task_registry, registered_task, arg_accessor);
  DynamicNodeAttrs node_attrs{
      /*task_type=*/i.node_attrs.task_type,
      /*device_coord=*/i.node_attrs.device_coord,
      /*mapping=*/i.node_attrs.mapping,
      /*op_attrs=*/i.node_attrs.op_attrs,
      /*layer_guid=*/i.node_attrs.layer_guid,
      /*per_device_op_state=*/per_device_op_state,
  };
  return DynamicNodeInvocation{
      /*inputs=*/
      i.inputs,
      /*node_attrs=*/
      node_attrs,
      /*outputs=*/
      i.outputs,
  };
}

InitializedComputationGraphInstance initialize_computation_graph_instance(
    ComputationGraphInstance const &instance,
    bidict<dynamic_tensor_guid_t, DynamicTensorAccessor> const &input_tensors,
    Allocator &allocator,
    LocalTaskRegistry &task_registry) {
  bidict<dynamic_tensor_guid_t, DynamicTensorAccessor> allocated_tensors =
      input_tensors;

  DynamicOpenDataflowGraph const &expanded_dg =
      instance.expanded_dataflow_graph;
  ASSERT(no_part_of_graph_is_initialized(expanded_dg));

  // Allocate all remaining tensors
  DynamicOpenDataflowGraph allocated_dg = transform_dynamic_invocation_set(
      expanded_dg, [&](DynamicNodeInvocation const &invocation) {
        auto allocate = [&](DynamicTensorSlot const &k,
                            DynamicValueAttrs const &v) {
          return std::pair{
              k,
              allocate_value(v, allocated_tensors, allocator),
          };
        };
        return DynamicNodeInvocation{
            /*inputs=*/
            transform(invocation.inputs, allocate),
            /*node_attrs=*/
            invocation.node_attrs,
            /*outputs=*/
            transform(invocation.outputs, allocate),
        };
      });

  // Initialize all operators and save the per-device op state
  DynamicOpenDataflowGraph initialized_dg = transform_dynamic_invocation_set(
      allocated_dg, [&](DynamicNodeInvocation const &invocation) {
        return initialize_node(invocation, task_registry);
      });

  ASSERT(graph_is_fully_initialized(initialized_dg));

  return InitializedComputationGraphInstance{
      initialized_dg, allocator, task_registry};
}

std::unordered_map<layer_guid_t, std::optional<milliseconds_t>>
    perform_forward_pass_for_computation_graph_instance(
        InitializedComputationGraphInstance const &instance) {

  NOT_IMPLEMENTED();
}

std::unordered_map<layer_guid_t, std::optional<milliseconds_t>>
    perform_backward_pass_for_computation_graph_instance(
        InitializedComputationGraphInstance const &instance) {

  NOT_IMPLEMENTED();
}

} // namespace FlexFlow
