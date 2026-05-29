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
#include "utils/bidict/algorithms/bidict_unordered_set_of.h"
#include "utils/containers/contains_key.h"
#include "utils/containers/flatmap.h"
#include "utils/containers/intersection.h"
#include "utils/containers/map_values2.h"
#include "utils/containers/set_difference.h"
#include "utils/containers/transform.h"
#include "utils/optional.h"
#include "task-spec/dynamic_graph/training_operation_attrs.h"

namespace FlexFlow {

bool node_is_copy(DynamicNodeAttrs const &n) {
  return n.op_attrs.has_value() && n.op_attrs.value().is_copy();
}

bool value_is_mapped(DynamicValueAttrs const &n) {
  return n.mapping.has_value();
}

bool no_part_of_graph_is_copy_inserted(DynamicOpenDataflowGraph const &g) {
  auto slot_is_mapped = [](DynamicTensorSlot const &) -> bool { return false; };
  for (DynamicNodeInvocation const &i : g.invocations) {
    if (node_is_copy(i.node_attrs)) {
      return false;
    }
    for (auto const &[slot, value] : i.inputs) {
      if (value_is_mapped(value)) {
        return false;
      }
    }
    for (auto const &[slot, value] : i.outputs) {
      if (value_is_mapped(value)) {
        return false;
      }
    }
  }
  return true;
}

bool graph_is_fully_copy_inserted(DynamicOpenDataflowGraph const &g) {
  auto node_is_any = [](DynamicNodeAttrs const &) -> bool { return true; };
  auto slot_is_mapped = [](DynamicTensorSlot const &) -> bool { return true; };

  return full_dynamic_graph_satisfies(
      g, node_is_any, value_is_mapped, slot_is_mapped);
}

void require_node_is_ready_for_copy_insertion(DynamicNodeAttrs const &node_attrs) {
  ASSERT(node_attrs.mapping.has_value());
}

void require_graph_is_ready_for_copy_insertion(DynamicOpenDataflowGraph const &g) {
  auto require_slot_is_ready_for_copy_insertion = [](DynamicTensorSlot const &slot) -> void {
    return;
  };

  auto require_value_is_ready_for_copy_insertion = [](DynamicValueAttrs const &value_attrs) -> void {
    return;
  };

  require_full_dynamic_graph_satisfies(
    g,
    require_node_is_ready_for_copy_insertion,
    require_value_is_ready_for_copy_insertion,
    require_slot_is_ready_for_copy_insertion);
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
      input_mapping = unstructured_relation_from_one_to_many(assert_unwrap(input.mapping));
  std::unordered_set<
      std::pair<ParallelTensorSpaceCoordinate, MachineSpaceCoordinate>>
      output_mapping = unstructured_relation_from_one_to_many(assert_unwrap(output.mapping));

  // Exclude the point shared between the input and output mappings, because
  // those will not result in actual copies once shard expansion is performed
  std::unordered_set<
      std::pair<ParallelTensorSpaceCoordinate, MachineSpaceCoordinate>>
      remove = intersection(input_mapping, output_mapping);

  DynamicValueAttrs filtered_input = input;
  filtered_input.mapping =
      one_to_many_from_unstructured_relation(set_difference(input_mapping, remove));

  DynamicValueAttrs filtered_output = output;
  filtered_output.mapping =
      one_to_many_from_unstructured_relation(set_difference(output_mapping, remove));

  return std::pair{filtered_input, filtered_output};
}

std::unordered_set<DynamicNodeInvocation> copies_for_invocation_inputs(
  DynamicNodeInvocation const &i,
  std::unordered_map<DynamicValueAttrs, DynamicValueAttrs> const &unmapped_value_to_src_mapped_value)
{
  if (training_op_attrs_has_op_type(assert_unwrap(i.node_attrs.op_attrs), OperatorType::REPLICATE)) {
    // copies should not be inserted before a replicate, as the replicate
    // implicitly includes the copy operations
    return {};
  }

  MappedOperatorTaskGroup mapping = assert_unwrap(i.node_attrs.mapping);

  auto map_tensor = [&](DynamicTensorSlot const &slot,
                        DynamicValueAttrs const &value) {
    return map_dynamic_value_attrs_for_task_group(slot, value, mapping);
  };

  std::map<DynamicTensorSlot, DynamicValueAttrs> mapped_inputs =
      map_values2(i.inputs, map_tensor);

  std::unordered_set<DynamicNodeInvocation> result;

  for (auto const &[slot, input] : i.inputs) {
    if (!contains_key(unmapped_value_to_src_mapped_value, input)) {
      continue;
    }

    DynamicValueAttrs src_mapped_value = unmapped_value_to_src_mapped_value.at(input);
    DynamicValueAttrs use_mapped_value = mapped_inputs.at(slot);

    if (src_mapped_value != use_mapped_value) {
      auto const &[filtered_source, filtered_use] = filter_mapping_to_avoid_degenerate_copies(src_mapped_value, use_mapped_value);
      DynamicNodeInvocation copy{
          /*inputs=*/{
              {
                  DynamicTensorSlot{
                    TensorSlotName::INPUT,
                    slot.slot_tensor_role,
                    /*task_shard=*/std::nullopt,
                  },
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
                  DynamicTensorSlot{
                    TensorSlotName::OUTPUT,
                    slot.slot_tensor_role,
                    /*task_shard=*/std::nullopt,
                  },
                  filtered_use,
              },
          },
      };
      result.insert(copy);
    }
  }

  return result;
}

std::unordered_set<DynamicNodeInvocation> perform_copy_insertion_for_invocation(
    DynamicNodeInvocation const &i,
    std::unordered_map<DynamicValueAttrs, DynamicValueAttrs> const
        &unmapped_value_to_mapped_source_value) {

  MappedOperatorTaskGroup mapping = assert_unwrap(i.node_attrs.mapping);

  auto map_tensor = [&](DynamicTensorSlot const &slot,
                        DynamicValueAttrs const &value) {
    return map_dynamic_value_attrs_for_task_group(slot, value, mapping);
  };

  DynamicNodeInvocation mapped_i = [&] {
    std::map<DynamicTensorSlot, DynamicValueAttrs> mapped_inputs =
        map_values2(i.inputs, map_tensor);
    std::map<DynamicTensorSlot, DynamicValueAttrs> mapped_outputs =
        map_values2(i.outputs, map_tensor);

    DynamicNodeInvocation r = i;
    r.inputs = mapped_inputs;
    r.outputs = mapped_outputs;
    return r;
  }();

  std::unordered_set<DynamicNodeInvocation> result = set_union(
    copies_for_invocation_inputs(i, unmapped_value_to_mapped_source_value),
    std::unordered_set<DynamicNodeInvocation>{
      mapped_i,
    });

  return result;
}

DynamicOpenDataflowGraph
    perform_copy_insertion(DynamicOpenDataflowGraph const &g) {

  ASSERT(no_part_of_graph_is_copy_inserted(g));
  require_graph_is_ready_for_copy_insertion(g);

  std::unordered_map<DynamicValueAttrs, DynamicValueAttrs>
      unmapped_value_to_mapped_source_value;
  for (DynamicNodeInvocation const &i : g.invocations) {
    for (auto const &[slot, value] : i.outputs) {
      unmapped_value_to_mapped_source_value.insert(
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
            return perform_copy_insertion_for_invocation(
                i, unmapped_value_to_mapped_source_value);
          }));

  ASSERT(graph_is_fully_copy_inserted(result));

  return result;
}

} // namespace FlexFlow
