#include "task-spec/dynamic_graph/pass_expansion.h"
#include "task-spec/dynamic_graph/dynamic_open_dataflow_graph.h"
#include "task-spec/dynamic_graph/dynamic_tensor_role.h"
#include "utils/containers/are_all_same.h"
#include "utils/containers/merge_disjoint_maps.h"
#include "utils/containers/transform.h"

namespace FlexFlow {

bool node_is_pass_expanded(DynamicNodeAttrs const &n) {
  return n.task_type.has_value();
}

bool slot_is_pass_expanded(DynamicTensorSlot const &s) {
  return s.slot_tensor_role.has_value();
}

bool value_is_pass_expanded(DynamicValueAttrs const &v) {
  return v.role.has_value();
}

bool no_part_of_graph_is_pass_expanded(DynamicOpenDataflowGraph const &g) {
  return no_part_of_dynamic_graph_satisfies(
      g, node_is_pass_expanded, value_is_pass_expanded, slot_is_pass_expanded);
}

bool graph_is_fully_pass_expanded(DynamicOpenDataflowGraph const &g) {
  return full_dynamic_graph_satisfies(
      g, node_is_pass_expanded, value_is_pass_expanded, slot_is_pass_expanded);
}

DynamicTensorSlot pass_expand_slot(DynamicTensorSlot const &s,
                                   FwbTensorType tensor_type) {
  ASSERT(s.slot_tensor_role == std::nullopt);

  DynamicTensorSlot result = s;
  result.slot_tensor_role =
      dynamic_tensor_role_from_fwb_tensor_type(tensor_type);
  return result;
}

DynamicValueAttrs pass_expand_value(DynamicValueAttrs const &v,
                                    FwbTensorType tensor_type) {
  ASSERT(!value_is_pass_expanded(v));

  DynamicValueAttrs result = v;
  result.role = DynamicTensorRole{tensor_type};
  return result;
};

DynamicNodeAttrs pass_expand_node(DynamicNodeAttrs const &n,
                                  DynamicTaskType task_type) {
  ASSERT(!node_is_pass_expanded(n));
  ASSERT(task_type == DynamicTaskType::FWD ||
         task_type == DynamicTaskType::BWD);

  DynamicNodeAttrs result = n;
  result.task_type = task_type;
  return result;
}

DynamicNodeInvocation perform_fwd_pass_expansion_for_invocation(
    DynamicNodeInvocation const &task) {

  auto to_fwd = [](DynamicTensorSlot const &k, DynamicValueAttrs const &v) {
    return std::pair{
        pass_expand_slot(k, FwbTensorType::FORWARD),
        pass_expand_value(v, FwbTensorType::FORWARD),
    };
  };

  return DynamicNodeInvocation{
      /*inputs=*/
      transform(task.inputs, to_fwd),
      /*node_attrs=*/
      pass_expand_node(task.node_attrs, DynamicTaskType::FWD),
      /*outputs=*/
      transform(task.outputs, to_fwd),
  };
}

DynamicNodeInvocation perform_bwd_pass_expansion_for_invocation(
    DynamicNodeInvocation const &invocation) {

  auto to_fwd = [](DynamicTensorSlot const &k, DynamicValueAttrs const &v) {
    return std::pair{
        pass_expand_slot(k, FwbTensorType::FORWARD),
        pass_expand_value(v, FwbTensorType::FORWARD),
    };
  };

  auto to_grad = [](DynamicTensorSlot const &k, DynamicValueAttrs const &v) {
    return std::pair{
        pass_expand_slot(k, FwbTensorType::GRADIENT),
        pass_expand_value(v, FwbTensorType::GRADIENT),
    };
  };

  return DynamicNodeInvocation{
      /*inputs=*/
      merge_disjoint_maps(std::vector{
          transform(invocation.inputs, to_fwd),
          transform(invocation.outputs, to_fwd),
          transform(invocation.outputs, to_grad),
      }),
      /*node_attrs=*/
      pass_expand_node(invocation.node_attrs, DynamicTaskType::BWD),
      /*outputs=*/
      transform(invocation.inputs, to_grad),
  };
}

DynamicOpenDataflowGraph
    perform_pass_expansion(DynamicOpenDataflowGraph const &g) {

  ASSERT(no_part_of_graph_is_pass_expanded(g));

  DynamicOpenDataflowGraph result = flatmap_dynamic_invocation_set(
      g, [](DynamicNodeInvocation const &invocation) {
        if (invocation.inputs.empty()) {
          return std::unordered_set{
              perform_fwd_pass_expansion_for_invocation(invocation),
          };
        } else {
          return std::unordered_set{
              perform_fwd_pass_expansion_for_invocation(invocation),
              perform_bwd_pass_expansion_for_invocation(invocation),
          };
        };
      });

  ASSERT(graph_is_fully_pass_expanded(result));

  return result;
}

} // namespace FlexFlow
