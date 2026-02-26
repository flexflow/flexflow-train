#include "task-spec/dynamic_graph/copy_insertion.h"
#include "op-attrs/parallel_tensor_space_coordinate.dtg.h"
#include "pcg/machine_space_coordinate.dtg.h"
#include "pcg/mapped_parallel_computation_graph/mapped_operator_task_group.h"
#include "task-spec/dynamic_graph/dynamic_node_attrs.dtg.h"
#include "task-spec/dynamic_graph/dynamic_node_invocation.dtg.h"
#include "task-spec/dynamic_graph/dynamic_open_dataflow_graph.h"
#include "task-spec/dynamic_graph/dynamic_tensor_slot.dtg.h"
#include "task-spec/dynamic_graph/dynamic_value_attrs.dtg.h"
#include "utils/containers/map_values2.h"
#include "utils/containers/transform.h"
#include "utils/optional.h"
#include <unordered_map>

namespace FlexFlow {

bool value_is_mapped(DynamicValueAttrs const &n) {
  return n.mapping.has_value();
}

bool no_part_of_graph_is_copy_inserted(DynamicOpenDataflowGraph const &g) {
  auto node_is_mapped = [](DynamicNodeAttrs const &) -> bool { return false; };
  auto slot_is_mapped = [](DynamicTensorSlot const &) -> bool { return false; };

  return no_part_of_dynamic_graph_satisfies(
      g, node_is_mapped, value_is_mapped, slot_is_mapped);
}

bool graph_is_fully_copy_inserted(DynamicOpenDataflowGraph const &g) {
  auto node_is_mapped = [](DynamicNodeAttrs const &) -> bool { return true; };
  auto slot_is_mapped = [](DynamicTensorSlot const &) -> bool { return true; };

  return full_dynamic_graph_satisfies(
      g, node_is_mapped, value_is_mapped, slot_is_mapped);
}

static DynamicValueAttrs map_dynamic_value_attrs_for_task_group(
    DynamicTensorSlot const &slot,
    DynamicValueAttrs const &value,
    MappedOperatorTaskGroup const &mapping) {
  DynamicValueAttrs result = value;
  result.mapping = get_tensor_bindings_for_slot_name(mapping, slot.slot_name);
  return result;
}

std::unordered_set<DynamicNodeInvocation> perform_copy_insertion_for_invocation(
    DynamicNodeInvocation const &i,
    std::unordered_map<DynamicValueAttrs, DynamicValueAttrs> const &sources) {

  MappedOperatorTaskGroup mapping = assert_unwrap(i.node_attrs.mapping);

  auto map_tensor = [&](DynamicTensorSlot const &slot,
                        DynamicValueAttrs const &value) {
    return map_dynamic_value_attrs_for_task_group(slot, value, mapping);
  };

  std::unordered_map<DynamicTensorSlot, DynamicValueAttrs> mapped_inputs =
      map_values2(i.inputs, map_tensor);
  std::unordered_map<DynamicTensorSlot, DynamicValueAttrs> mapped_outputs =
      map_values2(i.outputs, map_tensor);

  std::unordered_set<DynamicNodeInvocation> result{DynamicNodeInvocation{
      /*inputs=*/mapped_inputs,
      /*node_attrs=*/i.node_attrs,
      /*outputs=*/mapped_outputs,
  }};

  for (auto const &[slot, input] : i.inputs) {
    DynamicValueAttrs source_value = sources.at(input);
    DynamicValueAttrs use_value = mapped_inputs.at(slot);
    if (source_value != use_value) {
      result.insert(DynamicNodeInvocation{
          /*inputs=*/{{slot, source_value}},
          /*node_attrs=*/
          DynamicNodeAttrs{
              /*task_type=*/std::nullopt,
              /*device_coord=*/std::nullopt,
              /*mapping=*/std::nullopt,
              /*op_attrs*/ TrainingOperationAttrs{CopyAttrs{}},
              /*layer_guid=*/dynamic_layer_guid_t{dynamic_copy_layer_guid_t{}},
              /*per_device_op_state=*/std::nullopt,
          },
          /*outputs=*/{{slot, use_value}},
      });
    }
  }

  return result;
}

DynamicOpenDataflowGraph
    perform_copy_insertion(DynamicOpenDataflowGraph const &g) {

  ASSERT(no_part_of_graph_is_copy_inserted(g));

  std::unordered_map<DynamicValueAttrs, DynamicValueAttrs> sources;
  for (DynamicNodeInvocation const &i : g.invocations) {
    for (auto const &[slot, value] : i.outputs) {
      sources.insert(
          std::pair{value,
                    map_dynamic_value_attrs_for_task_group(
                        slot, value, assert_unwrap(i.node_attrs.mapping))});
    }
  }

  DynamicOpenDataflowGraph result =
      flatmap_dynamic_invocation_set(g, [&](DynamicNodeInvocation const &i) {
        return perform_copy_insertion_for_invocation(i, sources);
      });

  ASSERT(graph_is_fully_copy_inserted(result));

  return result;
}

} // namespace FlexFlow
