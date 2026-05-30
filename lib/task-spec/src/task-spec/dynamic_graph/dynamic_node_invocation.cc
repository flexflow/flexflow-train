#include "task-spec/dynamic_graph/dynamic_node_invocation.h"
#include "task-spec/dynamic_graph/dynamic_open_dataflow_graph.h"
#include "task-spec/dynamic_graph/training_operation_attrs.h"
#include "utils/containers/are_disjoint.h"
#include "utils/containers/set_union.h"
#include "utils/containers/unordered_set_of.h"
#include "utils/optional.h"

namespace FlexFlow {

std::unordered_map<DynamicTensorSlot, DynamicValueAttrs>
    get_slot_map_for_direction(DynamicNodeInvocation const &invocation,
                               TensorDirection direction) {
  switch (direction) {
    case TensorDirection::INCOMING:
      return invocation.inputs;
    case TensorDirection::OUTPUT:
      return invocation.outputs;
    default:
      PANIC("Unexpected direction {}", direction);
  }
}

TrainingOpType
    dynamic_node_invocation_get_op_type(DynamicNodeInvocation const &i) {
  TrainingOperationAttrs training_op_attrs =
      assert_unwrap(i.node_attrs.op_attrs);

  return training_op_attrs_get_op_type(training_op_attrs);
}

std::unordered_set<InternalDynamicSlotSite>
    get_dynamic_slot_sites_for_invocation(DynamicNodeInvocation const &i) {

  std::unordered_set<InternalDynamicSlotSite> input_slots =
      transform(unordered_set_of(i.inputs),
                [&](std::pair<DynamicTensorSlot, DynamicValueAttrs> const &p)
                    -> InternalDynamicSlotSite {
                  return InternalDynamicSlotSite{
                      /*invocation=*/i,
                      /*direction=*/TensorDirection::INCOMING,
                      /*slot_name=*/p.first,
                  };
                });

  std::unordered_set<InternalDynamicSlotSite> output_slots =
      transform(unordered_set_of(i.outputs),
                [&](std::pair<DynamicTensorSlot, DynamicValueAttrs> const &p)
                    -> InternalDynamicSlotSite {
                  return InternalDynamicSlotSite{
                      /*invocation=*/i,
                      /*direction=*/TensorDirection::OUTPUT,
                      /*slot_name=*/p.first,
                  };
                });

  ASSERT(are_disjoint(input_slots, output_slots));

  return set_union(input_slots, output_slots);
}

} // namespace FlexFlow
