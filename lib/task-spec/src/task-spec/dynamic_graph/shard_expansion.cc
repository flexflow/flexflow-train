#include "task-spec/dynamic_graph/shard_expansion.h"
#include "task-spec/dynamic_graph/dynamic_open_dataflow_graph.h"
#include "task-spec/dynamic_graph/dynamic_value_attrs.dtg.h"
#include "utils/bidict/algorithms/bidict_filter_keys.h"
#include "utils/containers/get_only.h"
#include "utils/containers/map_values2.h"
#include "utils/containers/require_same.h"
#include "utils/containers/transform.h"
#include "utils/optional.h"
#include "utils/containers/binary_merge_disjoint_maps.h"
#include "task-spec/dynamic_graph/dynamic_node_invocation.h"
#include "utils/containers/map_from_unordered.h"
#include "utils/one_to_many/one_to_many_filter_keys.h"

namespace FlexFlow {

bool node_is_shard_expanded(DynamicNodeAttrs const &n) {
  return n.device_coord.has_value();
}

bool node_is_ready_for_shard_expansion(DynamicNodeAttrs const &n) {
  if (!n.op_attrs.has_value()) {
    return false;
  }

  if (n.op_attrs.value().is_pcg_op()) {
    if (!n.mapping.has_value()) {
      return false;
    }
  }

  return true;
}

void require_node_is_ready_for_shard_expansion(DynamicNodeAttrs const &n) {
  ASSERT(n.op_attrs.has_value());
  if (n.op_attrs.value().is_pcg_op()) {
    ASSERT(n.mapping.has_value());
  }
}


bool invocation_is_fully_shard_expanded(DynamicNodeInvocation const &i) {
  auto slot_is_shard_expanded = [](DynamicTensorSlot const &) {
    return true;
  };

  return invocation_fully_satisfies(
    i,
    node_is_shard_expanded,
    value_is_shard_expanded,
    slot_is_shard_expanded);
}

bool value_is_shard_expanded(DynamicValueAttrs const &n) {
  return n.shard_coord.has_value() && n.mapping.has_value();
}

bool value_is_ready_for_shard_expansion(DynamicValueAttrs const &n) {
  return true;
}

void require_value_is_ready_for_shard_expansion(DynamicValueAttrs const &n) {
  return;
}

bool invocation_is_ready_for_shard_expansion(DynamicNodeInvocation const &i) {
  auto slot_is_ready_for_shard_expansion = [](DynamicTensorSlot const &) {
    return true;
  };

  return invocation_fully_satisfies(
    i,
    node_is_ready_for_shard_expansion,
    value_is_ready_for_shard_expansion,
    slot_is_ready_for_shard_expansion);
}

void require_invocation_is_ready_for_shard_expansion(DynamicNodeInvocation const &i) {
  auto require_slot_is_ready_for_shard_expansion = [](DynamicTensorSlot const &) -> void {
    return;
  };

  require_invocation_fully_satisfies(
    i,
    require_node_is_ready_for_shard_expansion,
    require_value_is_ready_for_shard_expansion,
    require_slot_is_ready_for_shard_expansion);
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

static OneToMany<ParallelTensorSpaceCoordinate, MachineSpaceCoordinate>
    restrict_tensor_mapping_keys_to_coord(
        OneToMany<ParallelTensorSpaceCoordinate, MachineSpaceCoordinate> const
            &mapping,
        ParallelTensorSpaceCoordinate const &parallel_tensor_coord) {
  return one_to_many_filter_keys(mapping, [&](ParallelTensorSpaceCoordinate const &p) {
    return p == parallel_tensor_coord;
  });
}

static DynamicNodeInvocationShardingInfo invocation_sharding_info_for_binding(
    DynamicNodeInvocation const &i,
    MachineSpaceCoordinate const &machine_coord,
    OperatorAtomicTaskShardBinding const &binding) {

  auto shard_expand_value_attrs =
      [&](DynamicTensorSlot const &s, DynamicValueAttrs const &v) -> DynamicValueAttrsShardingInfo {
    ParallelTensorSpaceCoordinate parallel_tensor_coord =
        binding.tensor_coords.at(s.slot_name);

    return DynamicValueAttrsShardingInfo{
      /*shard_coord=*/parallel_tensor_coord,
      /*mapping=*/restrict_tensor_mapping_keys_to_coord(v.mapping.value(), parallel_tensor_coord),
    };
  };

  DynamicNodeAttrs expanded_node_attrs = [&]() {
    DynamicNodeAttrs result = i.node_attrs;
    result.device_coord = machine_coord;
    return result;
  }();

  return DynamicNodeInvocationShardingInfo{
      /*device_coord=*/machine_coord,
      /*value_sharding=*/map_from_unordered(
        map_values2(
          binary_merge_disjoint_maps(i.inputs, i.outputs),
          shard_expand_value_attrs)),
  };
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
        [&](OneToMany<ParallelTensorSpaceCoordinate, MachineSpaceCoordinate> const &mapping)
          -> OneToMany<ParallelTensorSpaceCoordinate, MachineSpaceCoordinate>
        {
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

static std::set<DynamicNodeInvocationShardingInfo>
    generate_shard_expansion_for_copy(DynamicNodeInvocation const &i) {
  auto [input_slot, input] = get_only(i.inputs);
  auto [output_slot, output] = get_only(i.outputs);

  OneToMany<ParallelTensorSpaceCoordinate, MachineSpaceCoordinate> input_mapping =
      assert_unwrap(input.mapping);
  require_same(input_mapping.left_values(),
               assert_unwrap(output.mapping).left_values());

  return transform(
      input_mapping.left_values(),
      [&](ParallelTensorSpaceCoordinate const &p) -> DynamicNodeInvocationShardingInfo {
        // The machine coord for a copy is inherently nebulous because it
        // doesn't strictly run in any single location. Further, Realm has the
        // flexibility to issue a copy operation from anywhere in the machine,
        // including remotely. Here we choose machine_coord based on the input
        // because we expect this to align with the most efficient way to issue
        // copies in Realm, although the current Realm backend uses a
        // centralized controller and thus issues copies all from a single node.
        MachineSpaceCoordinate machine_coord = get_only(input_mapping.at_l(p));

        return invocation_sharding_info_for_binding(i,
                                            machine_coord,
                                            OperatorAtomicTaskShardBinding{{
                                                {input_slot.slot_name, p},
                                                {output_slot.slot_name, p},
                                            }});
      });
}

std::unordered_set<DynamicNodeInvocation>
    perform_shard_expansion_for_invocation(DynamicNodeInvocation const &i) {

  std::unordered_set<DynamicNodeInvocationShardingInfo>
    shard_expansion_info = generate_shard_expansion_for_invocation(i);

  return transform(
    shard_expansion_info,
    [&](DynamicNodeInvocationShardingInfo const &s)
      -> DynamicNodeInvocation
    {
      return apply_dynamic_node_invocation_sharding_info(i, s);
    });
}

bool graph_is_ready_for_shard_expansion(DynamicOpenDataflowGraph const &g) {
  auto slot_is_ready_for_shard_expansion = [](DynamicTensorSlot const &) -> bool {
    return false;
  };

  return full_dynamic_graph_satisfies(g,
                                      node_is_ready_for_shard_expansion,
                                      value_is_ready_for_shard_expansion,
                                      slot_is_ready_for_shard_expansion);
}


void require_graph_is_ready_for_shard_expansion(DynamicOpenDataflowGraph const &g) {
  auto require_slot_is_ready_for_shard_expansion = [](DynamicTensorSlot const &) -> void {
    return;
  };

  return require_full_dynamic_graph_satisfies(g,
                                              require_node_is_ready_for_shard_expansion,
                                              require_value_is_ready_for_shard_expansion,
                                              require_slot_is_ready_for_shard_expansion);
}

DynamicNodeAttrs apply_dynamic_node_attrs_sharding_info(
  DynamicNodeAttrs const &node_attrs,
  MachineSpaceCoordinate const &device_coord)
{
  DynamicNodeAttrs result = node_attrs;
  result.device_coord = device_coord;

  return result;
}

DynamicValueAttrs apply_dynamic_value_attrs_sharding_info(
  DynamicValueAttrs const &value_attrs,
  DynamicValueAttrsShardingInfo const &value_sharding_info)
{
  DynamicValueAttrs result = value_attrs;
  result.shard_coord = value_sharding_info.shard_coord;
  result.mapping = value_sharding_info.mapping;
  return result;
}

DynamicNodeInvocation apply_dynamic_node_invocation_sharding_info(
  DynamicNodeInvocation const &invocation,
  DynamicNodeInvocationShardingInfo const &invocation_sharding_info)
{
  require_invocation_is_ready_for_shard_expansion(invocation);

  auto shard_value = [&](DynamicTensorSlot const &slot, DynamicValueAttrs const &value_attrs) -> DynamicValueAttrs {
    DynamicValueAttrsShardingInfo sharding_info = invocation_sharding_info.value_sharding.at(slot);
    return apply_dynamic_value_attrs_sharding_info(value_attrs, sharding_info);
  };

  DynamicNodeInvocation result = DynamicNodeInvocation{
    /*inputs=*/map_values2(invocation.inputs, shard_value),
    /*node_attrs=*/apply_dynamic_node_attrs_sharding_info(invocation.node_attrs, invocation_sharding_info.device_coord),
    /*outputs=*/map_values2(invocation.outputs, shard_value),
  };

  ASSERT(invocation_is_fully_shard_expanded(result));
  return result;
}

std::unordered_set<DynamicNodeInvocationShardingInfo>
  generate_shard_expansion_for_invocation(DynamicNodeInvocation const &i)
{
  require_invocation_is_ready_for_shard_expansion(i);

  if (i.node_attrs.op_attrs.has_value() &&
      i.node_attrs.op_attrs.value().is_copy()) {
    return unordered_set_of(generate_shard_expansion_for_copy(i));
  }

  MappedOperatorTaskGroup mapping = assert_unwrap(i.node_attrs.mapping);

  std::unordered_set<MachineSpaceCoordinate> shard_machine_coords =
      mapping.get_shard_bindings().left_values();

  return transform(
      shard_machine_coords,
      [&](MachineSpaceCoordinate const &c) -> DynamicNodeInvocationShardingInfo {
        OperatorAtomicTaskShardBinding slot_bindings =
            mapping.get_shard_bindings().at_l(c);

        return invocation_sharding_info_for_binding(i, c, slot_bindings);
      });
}

DynamicOpenDataflowGraph
    perform_shard_expansion(DynamicOpenDataflowGraph const &g) {

  ASSERT(no_part_of_graph_is_shard_expanded(g));
  require_graph_is_ready_for_shard_expansion(g);

  DynamicOpenDataflowGraph result =
      flatmap_dynamic_invocation_set(g, [&](DynamicNodeInvocation const &i) {
        return perform_shard_expansion_for_invocation(i);
      });

  ASSERT(graph_is_fully_shard_expanded(result));

  return result;
}

} // namespace FlexFlow
