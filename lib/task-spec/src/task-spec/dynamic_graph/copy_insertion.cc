#include "task-spec/dynamic_graph/copy_insertion.h"
#include "op-attrs/parallel_tensor_space_coordinate.dtg.h"
#include "op-attrs/tensor_slot_name.dtg.h"
#include "pcg/machine_space_coordinate.dtg.h"
#include "pcg/mapped_parallel_computation_graph/mapped_operator_task_group.h"
#include "task-spec/dynamic_graph/dynamic_node_attrs.dtg.h"
#include "task-spec/dynamic_graph/dynamic_node_invocation.dtg.h"
#include "task-spec/dynamic_graph/dynamic_open_dataflow_graph.h"
#include "task-spec/dynamic_graph/dynamic_task_type.h"
#include "task-spec/dynamic_graph/dynamic_tensor_slot.dtg.h"
#include "task-spec/dynamic_graph/dynamic_value_attrs.dtg.h"
#include "utils/bidict/algorithms/bidict_from_pairs.h"
#include "utils/bidict/algorithms/unordered_set_of.h"
#include "utils/containers/contains_key.h"
#include "utils/containers/flatmap.h"
#include "utils/containers/intersection.h"
#include "utils/containers/map_values2.h"
#include "utils/containers/set_difference.h"
#include "utils/containers/transform.h"
#include "utils/optional.h"

namespace FlexFlow {

bool node_is_copy(DynamicNodeAttrs const &n) {
  return n.op_attrs.has_value() && n.op_attrs.value().is_copy();
}

bool value_is_mapped(DynamicValueAttrs const &n) {
  return n.mapping.has_value();
}

bool no_part_of_graph_is_copy_inserted(DynamicOpenDataflowGraph const &g) {
  auto slot_is_mapped = [](DynamicTensorSlot const &) -> bool { return false; };

  return no_part_of_dynamic_graph_satisfies(
      g, node_is_copy, value_is_mapped, slot_is_mapped);
}

bool graph_is_fully_copy_inserted(DynamicOpenDataflowGraph const &g) {
  auto node_is_any = [](DynamicNodeAttrs const &) -> bool { return true; };
  auto slot_is_mapped = [](DynamicTensorSlot const &) -> bool { return true; };

  return full_dynamic_graph_satisfies(
      g, node_is_any, value_is_mapped, slot_is_mapped);
}

static DynamicValueAttrs map_dynamic_value_attrs_for_task_group(
    DynamicTensorSlot const &slot,
    DynamicValueAttrs const &value,
    MappedOperatorTaskGroup const &mapping) {
  DynamicValueAttrs result = value;
  result.mapping = get_tensor_bindings_for_slot_name(mapping, slot.slot_name);
  return result;
}

static std::pair<DynamicValueAttrs, DynamicValueAttrs>
    filter_mapping_to_avoid_degenerate_copies(DynamicValueAttrs const &input,
                                              DynamicValueAttrs const &output) {
  std::unordered_set<
      std::pair<ParallelTensorSpaceCoordinate, MachineSpaceCoordinate>>
      input_mapping = unordered_set_of(assert_unwrap(input.mapping));
  std::unordered_set<
      std::pair<ParallelTensorSpaceCoordinate, MachineSpaceCoordinate>>
      output_mapping = unordered_set_of(assert_unwrap(output.mapping));

  // Exclude the point shared between the input and output mappings, because
  // those will not result in actual copies once shard expansion is performed
  std::unordered_set<
      std::pair<ParallelTensorSpaceCoordinate, MachineSpaceCoordinate>>
      remove = intersection(input_mapping, output_mapping);

  DynamicValueAttrs filtered_input = input;
  filtered_input.mapping =
      bidict_from_pairs(set_difference(input_mapping, remove));

  DynamicValueAttrs filtered_output = output;
  filtered_output.mapping =
      bidict_from_pairs(set_difference(output_mapping, remove));

  return std::pair{filtered_input, filtered_output};
}

std::unordered_set<DynamicNodeInvocation> perform_copy_insertion_for_invocation(
    DynamicNodeInvocation const &i,
    std::unordered_map<DynamicValueAttrs, DynamicValueAttrs> const
        &mapped_source_value) {

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
    if (!contains_key(mapped_source_value, input)) {
      continue;
    }

    DynamicValueAttrs source_value = mapped_source_value.at(input);
    DynamicValueAttrs use_value = mapped_inputs.at(slot);
    if (source_value != use_value) {
      auto const &[filtered_source, filtered_use] =
          filter_mapping_to_avoid_degenerate_copies(source_value, use_value);
      DynamicNodeInvocation copy{
          /*inputs=*/{
              {
                  DynamicTensorSlot{TensorSlotName::INPUT,
                                    slot.slot_tensor_role},
                  filtered_source,
              },
          },
          /*node_attrs=*/
          DynamicNodeAttrs{
              /*task_type=*/transform(
                  slot.slot_tensor_role,
                  dynamic_task_type_from_tensor_role_for_copy),
              /*device_coord=*/std::nullopt,
              /*mapping=*/std::nullopt,
              /*op_attrs*/ TrainingOperationAttrs{CopyAttrs{}},
              /*layer_guid=*/dynamic_layer_guid_t{dynamic_copy_layer_guid_t{}},
              /*per_device_op_state=*/std::nullopt,
          },
          /*outputs=*/
          {
              {
                  DynamicTensorSlot{TensorSlotName::OUTPUT,
                                    slot.slot_tensor_role},
                  filtered_use,
              },
          },
      };
      result.insert(copy);
    }
  }

  return result;
}

DynamicOpenDataflowGraph
    perform_copy_insertion(DynamicOpenDataflowGraph const &g) {

  ASSERT(no_part_of_graph_is_copy_inserted(g));

  std::unordered_map<DynamicValueAttrs, DynamicValueAttrs> mapped_source_value;
  for (DynamicNodeInvocation const &i : g.invocations) {
    for (auto const &[slot, value] : i.outputs) {
      mapped_source_value.insert(
          std::pair{value,
                    map_dynamic_value_attrs_for_task_group(
                        slot, value, assert_unwrap(i.node_attrs.mapping))});
    }
  }

  // Use regular flatmap here to remove duplicates (we don't want to copy the
  // same tensor to the same place multiple times)
  DynamicOpenDataflowGraph result =
      dynamic_open_dataflow_graph_from_invocation_set(
          flatmap(g.invocations, [&](DynamicNodeInvocation const &i) {
            return perform_copy_insertion_for_invocation(i,
                                                         mapped_source_value);
          }));

  ASSERT(graph_is_fully_copy_inserted(result));

  return result;
}

} // namespace FlexFlow
