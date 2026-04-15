#include "task-spec/dynamic_graph/shard_expansion.h"
#include "task-spec/dynamic_graph/dynamic_open_dataflow_graph.h"
#include "task-spec/dynamic_graph/dynamic_value_attrs.dtg.h"
#include "utils/bidict/algorithms/filter_keys.h"
#include "utils/containers/get_only.h"
#include "utils/containers/map_values2.h"
#include "utils/containers/require_same.h"
#include "utils/containers/transform.h"
#include "utils/optional.h"

namespace FlexFlow {

bool node_is_shard_expanded(DynamicNodeAttrs const &n) {
  return n.device_coord.has_value();
}

bool value_is_shard_expanded(DynamicValueAttrs const &n) {
  return n.shard_coord.has_value();
}

bool no_part_of_graph_is_shard_expanded(DynamicOpenDataflowGraph const &g) {
  auto slot_is_shard_expanded = [](DynamicTensorSlot const &) -> bool {
    return false;
  };

  return no_part_of_dynamic_graph_satisfies(g,
                                            node_is_shard_expanded,
                                            value_is_shard_expanded,
                                            slot_is_shard_expanded);
}

bool graph_is_fully_shard_expanded(DynamicOpenDataflowGraph const &g) {
  auto slot_is_shard_expanded = [](DynamicTensorSlot const &) -> bool {
    return true;
  };

  return full_dynamic_graph_satisfies(g,
                                      node_is_shard_expanded,
                                      value_is_shard_expanded,
                                      slot_is_shard_expanded);
}
static bidict<ParallelTensorSpaceCoordinate, MachineSpaceCoordinate>
    restrict_tensor_mapping_keys_to_coord(
        bidict<ParallelTensorSpaceCoordinate, MachineSpaceCoordinate> const
            &mapping,
        ParallelTensorSpaceCoordinate const &parallel_tensor_coord) {
  return filter_keys(mapping, [&](ParallelTensorSpaceCoordinate const &p) {
    return p == parallel_tensor_coord;
  });
}

static DynamicNodeInvocation shard_invocation_for_binding(
    DynamicNodeInvocation const &i,
    MachineSpaceCoordinate const &machine_coord,
    OperatorAtomicTaskShardBinding const &binding) {
  auto shard_expand_value_attrs =
      [&](DynamicTensorSlot const &s,
          DynamicValueAttrs const &v) -> DynamicValueAttrs {
    ParallelTensorSpaceCoordinate parallel_tensor_coord =
        binding.tensor_coords.at(s.slot_name);

    DynamicValueAttrs result = v;
    result.shard_coord = parallel_tensor_coord;
    result.mapping = transform(
        v.mapping,
        [&](bidict<ParallelTensorSpaceCoordinate, MachineSpaceCoordinate> const
                &mapping) {
          return restrict_tensor_mapping_keys_to_coord(mapping,
                                                       parallel_tensor_coord);
        });
    return result;
  };

  DynamicNodeAttrs expanded_node_attrs = [&]() {
    DynamicNodeAttrs result = i.node_attrs;
    result.device_coord = machine_coord;
    return result;
  }();

  return DynamicNodeInvocation{
      /*inputs=*/map_values2(i.inputs, shard_expand_value_attrs),
      /*node_attrs=*/expanded_node_attrs,
      /*outputs=*/map_values2(i.outputs, shard_expand_value_attrs),
  };
}

static std::unordered_set<DynamicNodeInvocation>
    perform_shard_expansion_for_replicate(DynamicNodeInvocation const &i) {
  auto const &[input_slot, input] = get_only(i.inputs);
  auto const &[output_slot, output] = get_only(i.outputs);

  bidict<ParallelTensorSpaceCoordinate, MachineSpaceCoordinate> input_mapping =
      assert_unwrap(input.mapping);
  bidict<ParallelTensorSpaceCoordinate, MachineSpaceCoordinate> output_mapping =
      assert_unwrap(output.mapping);

  return transform(output_mapping.left_values(),
                   [&](ParallelTensorSpaceCoordinate const &p) {
                     ParallelTensorSpaceCoordinate input_p{
                         /*sum_component=*/p.sum_component,
                         /*discard_copy_component=*/nonnegative_int{0},
                         /*shard_components=*/p.shard_components,
                     };
                     return shard_invocation_for_binding(
                         i,
                         output_mapping.at_l(p),
                         OperatorAtomicTaskShardBinding{{
                             {input_slot.slot_name, input_p},
                             {output_slot.slot_name, p},
                         }});
                   });
}

static std::unordered_set<DynamicNodeInvocation>
    perform_shard_expansion_for_replicate_bwd(DynamicNodeInvocation const &i) {

  std::optional<DynamicValueAttrs> output_grad_opt;
  std::optional<DynamicValueAttrs> output_fwd_opt;
  std::optional<DynamicTensorSlot> output_grad_slot_opt;
  std::optional<DynamicTensorSlot> output_fwd_slot_opt;

  for (auto const &[slot, value] : i.inputs) {
    if (slot.slot_tensor_role == DynamicTensorRole{FwbTensorType::GRADIENT}) {
      output_grad_slot_opt = slot;
      output_grad_opt = value;
    } else {
      output_fwd_slot_opt = slot;
      output_fwd_opt = value;
    }
  }

  DynamicValueAttrs output_grad = assert_unwrap(output_grad_opt);
  DynamicValueAttrs output_fwd = assert_unwrap(output_fwd_opt);
  DynamicTensorSlot output_grad_slot = assert_unwrap(output_grad_slot_opt);
  DynamicTensorSlot output_fwd_slot = assert_unwrap(output_fwd_slot_opt);
  auto const &[input_grad_slot, input_grad] = get_only(i.outputs);

  bidict<ParallelTensorSpaceCoordinate, MachineSpaceCoordinate>
      output_grad_mapping = assert_unwrap(output_grad.mapping);
  bidict<ParallelTensorSpaceCoordinate, MachineSpaceCoordinate>
      input_grad_mapping = assert_unwrap(input_grad.mapping);

  std::unordered_map<FFOrdered<nonnegative_int>,
                     std::unordered_set<ParallelTensorSpaceCoordinate>>
      by_shard;
  for (auto const &p : output_grad_mapping.left_values()) {
    by_shard[p.shard_components].insert(p);
  }

  std::unordered_set<DynamicNodeInvocation> result;
  for (auto const &[shard_components, replica_coords] : by_shard) {
    ParallelTensorSpaceCoordinate src_p{
        nonnegative_int{0}, nonnegative_int{0}, shard_components};
    MachineSpaceCoordinate src_machine = input_grad_mapping.at_l(src_p);

    bidict<ParallelTensorSpaceCoordinate, MachineSpaceCoordinate>
        replica_mapping;
    for (auto const &p : replica_coords) {
      replica_mapping.equate(p, output_grad_mapping.at_l(p));
    }

    DynamicValueAttrs sharded_output_grad = output_grad;
    sharded_output_grad.mapping = replica_mapping;
    sharded_output_grad.shard_coord = src_p;

    DynamicValueAttrs sharded_output_fwd = output_fwd;
    sharded_output_fwd.mapping = replica_mapping;
    sharded_output_fwd.shard_coord = src_p;

    DynamicValueAttrs sharded_input_grad = input_grad;
    sharded_input_grad.mapping =
        bidict<ParallelTensorSpaceCoordinate, MachineSpaceCoordinate>{
            {src_p, src_machine}};
    sharded_input_grad.shard_coord = src_p;

    DynamicNodeAttrs sharded_node = i.node_attrs;
    sharded_node.device_coord = src_machine;

    result.insert(DynamicNodeInvocation{
        /*inputs=*/{
            {output_fwd_slot, sharded_output_fwd},
            {output_grad_slot, sharded_output_grad},
        },
        /*node_attrs=*/sharded_node,
        /*outputs=*/
        {
            {input_grad_slot, sharded_input_grad},
        },
    });
  }
  return result;
}

static std::unordered_set<DynamicNodeInvocation>
    perform_shard_expansion_for_copy(DynamicNodeInvocation const &i) {
  auto [input_slot, input] = get_only(i.inputs);
  auto [output_slot, output] = get_only(i.outputs);
  bidict<ParallelTensorSpaceCoordinate, MachineSpaceCoordinate> input_mapping =
      assert_unwrap(input.mapping);
  require_same(input_mapping.left_values(),
               assert_unwrap(output.mapping).left_values());

  return transform(
      input_mapping.left_values(), [&](ParallelTensorSpaceCoordinate const &p) {
        // The machine coord for a copy is inherently nebulous because it
        // doesn't strictly run in any single location. Further, Realm has the
        // flexibility to issue a copy operation from anywhere in the machine,
        // including remotely. Here we choose machine_coord based on the input
        // because we expect this to align with the most efficient way to issue
        // copies in Realm, although the current Realm backend uses a
        // centralized controller and thus issues copies all from a single node.
        MachineSpaceCoordinate machine_coord = input_mapping.at_l(p);

        return shard_invocation_for_binding(i,
                                            machine_coord,
                                            OperatorAtomicTaskShardBinding{{
                                                {input_slot.slot_name, p},
                                                {output_slot.slot_name, p},
                                            }});
      });
}

std::unordered_set<DynamicNodeInvocation>
    perform_shard_expansion_for_invocation(DynamicNodeInvocation const &i) {
  if (i.node_attrs.op_attrs.has_value() &&
      i.node_attrs.op_attrs.value().is_copy()) {
    return perform_shard_expansion_for_copy(i);
  }

  bool const is_replicate =
      i.node_attrs.op_attrs.has_value() &&
      i.node_attrs.op_attrs.value().has<PCGOperatorAttrs>() &&
      i.node_attrs.op_attrs.value()
          .get<PCGOperatorAttrs>()
          .has<ReplicateAttrs>();

  // forward replicate
  if (is_replicate && i.node_attrs.task_type.has_value() &&
      i.node_attrs.task_type.value() == DynamicTaskType::FWD) {
    return perform_shard_expansion_for_replicate(i);
  }

  // backward replicate
  if (is_replicate && i.node_attrs.task_type.has_value() &&
      i.node_attrs.task_type.value() == DynamicTaskType::BWD) {
    return perform_shard_expansion_for_replicate_bwd(i);
  }

  MappedOperatorTaskGroup mapping = assert_unwrap(i.node_attrs.mapping);

  std::unordered_set<MachineSpaceCoordinate> shard_machine_coords =
      mapping.get_shard_bindings().left_values();

  return transform(
      shard_machine_coords,
      [&](MachineSpaceCoordinate const &c) -> DynamicNodeInvocation {
        OperatorAtomicTaskShardBinding slot_bindings =
            mapping.get_shard_bindings().at_l(c);

        return shard_invocation_for_binding(i, c, slot_bindings);
      });
}

DynamicOpenDataflowGraph
    perform_shard_expansion(DynamicOpenDataflowGraph const &g) {

  ASSERT(no_part_of_graph_is_shard_expanded(g));

  DynamicOpenDataflowGraph result =
      flatmap_dynamic_invocation_set(g, [&](DynamicNodeInvocation const &i) {
        return perform_shard_expansion_for_invocation(i);
      });

  ASSERT(graph_is_fully_shard_expanded(result));

  return result;
}

} // namespace FlexFlow
