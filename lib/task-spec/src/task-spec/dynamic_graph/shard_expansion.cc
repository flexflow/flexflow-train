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
#include "task-spec/dynamic_graph/training_operation_attrs.h"
#include "utils/one_to_many/require_one_to_many_is_bijection.h"
#include "pcg/mapped_parallel_computation_graph/operator_atomic_task_shard_binding.h"
#include "utils/bidict/algorithms/bidict_filter_values.h"
#include "task-spec/dynamic_graph/dynamic_tensor_role.h"
#include "utils/containers/merge_disjoint_maps.h"
#include "utils/containers/map_keys.h"
#include "utils/containers/require_only_key.h"

namespace FlexFlow {

bool node_is_shard_expanded(DynamicNodeAttrs const &n) {
  return n.device_coords.has_value();
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
    result.device_coords = nonempty_set{machine_coord};
    return result;
  }();

  return DynamicNodeInvocationShardingInfo{
      /*device_coord=*/nonempty_set{machine_coord},
      /*value_sharding=*/map_values2(
          binary_merge_disjoint_maps(i.inputs, i.outputs),
          shard_expand_value_attrs),
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
    result.device_coords = nonempty_set{machine_coord};
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

static std::set<DynamicNodeInvocationShardingInfo>
    generate_shard_expansion_for_fwd_replicate(DynamicNodeInvocation const &i) {
  ASSERT(i.node_attrs.task_type == DynamicTaskType::FWD);

  MappedOperatorTaskGroup node_mapping = assert_unwrap(i.node_attrs.mapping);

  DynamicTensorSlot expected_input_slot = DynamicTensorSlot{
    /*slot_name=*/TensorSlotName::INPUT,
    /*slot_tensor_role=*/mk_dynamic_tensor_role_fwd(),
    /*task_shard=*/std::nullopt,
  };

  DynamicValueAttrs input = require_only_key(i.inputs, expected_input_slot);

  DynamicTensorSlot expected_output_slot = DynamicTensorSlot{
    /*slot_name=*/TensorSlotName::OUTPUT,
    /*slot_tensor_role=*/mk_dynamic_tensor_role_fwd(),
    /*task_shard=*/std::nullopt,
  };

  DynamicValueAttrs output = require_only_key(i.outputs, expected_output_slot);

  bidict<ParallelTensorSpaceCoordinate, MachineSpaceCoordinate>
    input_value_mapping = require_one_to_many_is_bijection(
      assert_unwrap(input.mapping));

  std::set<ParallelTensorSpaceCoordinate> input_tensor_shards = set_of(input_value_mapping.left_values());
  
  bidict<ParallelTensorSpaceCoordinate, MachineSpaceCoordinate>
    output_value_mapping = require_one_to_many_is_bijection(
      assert_unwrap(output.mapping));

  auto get_task_shard_machine_coords_for_input_tensor_shard
    = [&](ParallelTensorSpaceCoordinate const &input_tensor_shard)
        -> nonempty_set<MachineSpaceCoordinate>
  {
    bidict<MachineSpaceCoordinate, OperatorAtomicTaskShardBinding> dependent_on_input_tensor_shard
      = bidict_filter_values(
        node_mapping.get_shard_bindings(),
        [&](OperatorAtomicTaskShardBinding const &b) -> bool {
          return ptensor_space_coord_for_slot_name(b, TensorSlotName::INPUT) == input_tensor_shard;
        });

    return nonempty_set(set_of(dependent_on_input_tensor_shard.left_values()));
  };

  auto invocation_sharding_info_for_input_tensor_shard = [&](ParallelTensorSpaceCoordinate const &c)
    -> DynamicNodeInvocationShardingInfo
  {
    nonempty_set<MachineSpaceCoordinate> task_shard_machine_coords =
      get_task_shard_machine_coords_for_input_tensor_shard(c);

    std::map<MachineSpaceCoordinate, DynamicValueAttrsShardingInfo> output_sharding_infos =
      generate_map(task_shard_machine_coords.unwrap_as_set(),
                   [&](MachineSpaceCoordinate const &mc)
                     -> DynamicValueAttrsShardingInfo
                   {
                     ParallelTensorSpaceCoordinate pc = output_value_mapping.at_r(mc);

                     return DynamicValueAttrsShardingInfo{
                       /*shard_coord=*/pc,
                       /*mapping=*/OneToMany<ParallelTensorSpaceCoordinate, MachineSpaceCoordinate>{
                         {
                           pc,
                           {mc},
                         },
                       },
                     };
                   });

    std::map<DynamicTensorSlot, DynamicValueAttrsShardingInfo> keyed_output_sharding_infos =
      map_keys(output_sharding_infos,
               [&](MachineSpaceCoordinate const &mc) -> DynamicTensorSlot {
                 return DynamicTensorSlot{
                   /*slot_name=*/TensorSlotName::OUTPUT,
                   /*slot_tensor_role=*/mk_dynamic_tensor_role_fwd(),
                   /*task_shard=*/mc,
                 };
               });

    DynamicTensorSlot input_slot = DynamicTensorSlot{
      /*slot_name=*/TensorSlotName::INPUT,
      /*slot_tensor_role=*/mk_dynamic_tensor_role_fwd(),
      /*task_shard=*/std::nullopt,
    };

    DynamicValueAttrsShardingInfo input_sharding_info = DynamicValueAttrsShardingInfo{
      /*shard_coord=*/c,
      /*mapping=*/OneToMany<ParallelTensorSpaceCoordinate, MachineSpaceCoordinate>{
        {
          c,
          {input_value_mapping.at_l(c)},
        },
      },
    };

    std::map<DynamicTensorSlot, DynamicValueAttrsShardingInfo> sharding_infos = 
      binary_merge_disjoint_maps(
        keyed_output_sharding_infos,
        std::map<DynamicTensorSlot, DynamicValueAttrsShardingInfo>{
          {
            input_slot,
            input_sharding_info,
          },
        });

    return DynamicNodeInvocationShardingInfo{
      /*device_coords=*/task_shard_machine_coords,
      /*value_sharding=*/sharding_infos,
    };
  };

  return transform(input_tensor_shards, invocation_sharding_info_for_input_tensor_shard);
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
  nonempty_set<MachineSpaceCoordinate> const &device_coords)
{
  DynamicNodeAttrs result = node_attrs;
  result.device_coords = device_coords;

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

  auto shard_value = [&](DynamicTensorSlot const &slot, DynamicValueAttrs const &value_attrs)
      -> DynamicValueAttrs
  {
    DynamicValueAttrsShardingInfo sharding_info = invocation_sharding_info.value_sharding.at(slot);
    return apply_dynamic_value_attrs_sharding_info(value_attrs, sharding_info);
  };

  DynamicNodeInvocation result = DynamicNodeInvocation{
    /*inputs=*/map_values2(invocation.inputs, shard_value),
    /*node_attrs=*/apply_dynamic_node_attrs_sharding_info(
      invocation.node_attrs, invocation_sharding_info.device_coords),
    /*outputs=*/map_values2(invocation.outputs, shard_value),
  };

  ASSERT(invocation_is_fully_shard_expanded(result));
  return result;
}

std::unordered_set<DynamicNodeInvocationShardingInfo>
  generate_shard_expansion_for_invocation(DynamicNodeInvocation const &i)
{
  require_invocation_is_ready_for_shard_expansion(i);

  if (i.node_attrs.op_attrs.value().is_copy()) {
    return unordered_set_of(generate_shard_expansion_for_copy(i));
  }

  if (training_op_attrs_has_op_type(i.node_attrs.op_attrs.value(), OperatorType::REPLICATE)) {
    return unordered_set_of(generate_shard_expansion_for_fwd_replicate(i));
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
