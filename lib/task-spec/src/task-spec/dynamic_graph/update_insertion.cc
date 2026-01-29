#include "task-spec/dynamic_graph/update_insertion.h"
#include "pcg/optimizer_attrs.h"
#include "task-spec/dynamic_graph/dynamic_open_dataflow_graph.h"
#include "task-spec/dynamic_graph/dynamic_tensor_role.h"
#include "task-spec/dynamic_graph/dynamic_tensor_slot.h"
#include "task-spec/dynamic_graph/dynamic_value_attrs.h"
#include "task-spec/optimizer.h"
#include "utils/containers/get_only.h"
#include "utils/containers/map_from_pairs.h"
#include "utils/containers/set_union.h"

namespace FlexFlow {

static std::pair<DynamicTensorSlot, DynamicValueAttrs>
    get_weight_output(DynamicNodeInvocation const &i) {
  ASSERT(i.node_attrs.op_attrs.value().is_weight());
  ASSERT(i.inputs.size() == 0);

  auto [slot, value_attrs] = get_only(i.outputs);

  return std::pair{
      slot,
      value_attrs,
  };
}

static DynamicTensorSlot tensor_slot_with_role(DynamicTensorSlot const &slot,
                                               DynamicTensorRole role) {
  DynamicTensorSlot result = slot;
  result.slot_tensor_role = role;

  return result;
}

static DynamicValueAttrs
    dynamic_value_attrs_with_role(DynamicValueAttrs const &attrs,
                                  DynamicTensorRole role) {
  DynamicValueAttrs result = attrs;
  result.role = role;

  return result;
}

static DynamicNodeInvocation get_update_invocation_for_invocation(
    DynamicNodeInvocation const &i, OptimizerAttrs const &optimizer_attrs) {

  auto output = get_weight_output(i);
  DynamicTensorSlot slot = output.first;
  DynamicValueAttrs value_attrs = output.second;

  ASSERT(value_attrs.accessor == std::nullopt);

  DynamicNodeAttrs update_node_attrs = i.node_attrs;
  update_node_attrs.task_type = DynamicTaskType::UPD;

  auto create_binding_for_role = [&](DynamicTensorRole const &role)
      -> std::pair<DynamicTensorSlot, DynamicValueAttrs> {
    DynamicTensorSlot binding_slot = tensor_slot_with_role(slot, role);
    DynamicValueAttrs binding_attrs = dynamic_value_attrs_with_role(
        value_attrs, mk_dynamic_tensor_role_fwd());

    return std::pair{
        binding_slot,
        binding_attrs,
    };
  };

  std::unordered_set<DynamicTensorRole> tensor_roles = set_union(
      std::unordered_set{
          mk_dynamic_tensor_role_fwd(),
          mk_dynamic_tensor_role_bwd(),
      },
      transform(get_slot_names_for_optimizer(optimizer_attrs),
                mk_dynamic_tensor_role_opt));

  return DynamicNodeInvocation{
      /*inputs=*/map_from_pairs(
          transform(tensor_roles, create_binding_for_role)),
      /*node_attrs=*/update_node_attrs,
      /*outputs=*/std::unordered_map<DynamicTensorSlot, DynamicValueAttrs>{},
  };
}

std::unordered_set<DynamicNodeInvocation>
    perform_update_insertion_for_invocation(
        DynamicNodeInvocation const &invocation,
        OptimizerAttrs const &optimizer_attrs) {

  if (invocation.node_attrs.task_type.value() == DynamicTaskType::FWD &&
      invocation.node_attrs.op_attrs.value().is_weight()) {
    return std::unordered_set{
        invocation,
        get_update_invocation_for_invocation(invocation, optimizer_attrs),
    };
  } else {
    return std::unordered_set{
        invocation,
    };
  };
}

DynamicOpenDataflowGraph
    perform_update_insertion(DynamicOpenDataflowGraph const &g,
                             OptimizerAttrs const &optimizer_attrs) {

  return flatmap_dynamic_invocation_set(g, [&](DynamicNodeInvocation const &i) {
    return perform_update_insertion_for_invocation(i, optimizer_attrs);
  });
}

} // namespace FlexFlow
