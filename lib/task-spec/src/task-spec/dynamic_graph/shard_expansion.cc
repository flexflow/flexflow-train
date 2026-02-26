#include "task-spec/dynamic_graph/shard_expansion.h"
#include "task-spec/dynamic_graph/dynamic_open_dataflow_graph.h"
#include "utils/bidict/algorithms/filter_keys.h"
#include "utils/containers/map_values2.h"
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
    subset_tensor_mapping_for_coord(
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
          return subset_tensor_mapping_for_coord(mapping,
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

std::unordered_set<DynamicNodeInvocation>
    perform_shard_expansion_for_invocation(DynamicNodeInvocation const &i) {

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
