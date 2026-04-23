#include "task-spec/dynamic_graph/shard_expansion.h"
#include "task-spec/dynamic_graph/dynamic_open_dataflow_graph.h"
#include "task-spec/dynamic_graph/dynamic_value_attrs.dtg.h"
#include "task-spec/dynamic_graph/parallel_op_utils.h"
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

static bool has_task_type(DynamicNodeAttrs const &n, DynamicTaskType t) {
  return n.task_type.has_value() && n.task_type.value() == t;
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
    perform_shard_expansion_one_to_many(
        DynamicNodeInvocation const &i,
        std::function<ParallelTensorSpaceCoordinate(
            ParallelTensorSpaceCoordinate const &)> output_to_input_coord) {

  if (has_task_type(i.node_attrs, DynamicTaskType::FWD)) {
    auto const &[input_slot, input] = get_only(i.inputs);
    auto const &[output_slot, output] = get_only(i.outputs);

    bidict<ParallelTensorSpaceCoordinate, MachineSpaceCoordinate>
        output_mapping = assert_unwrap(output.mapping);

    return transform(output_mapping.left_values(),
                     [&](ParallelTensorSpaceCoordinate const &p) {
                       ParallelTensorSpaceCoordinate input_p =
                           output_to_input_coord(p);
                       return shard_invocation_for_binding(
                           i,
                           output_mapping.at_l(p),
                           OperatorAtomicTaskShardBinding{{
                               {input_slot.slot_name, input_p},
                               {output_slot.slot_name, p},
                           }});
                     });
  }

  // BWD case — inputs are OUTPUT/FWD and OUTPUT/GRAD, output is INPUT/GRAD
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
      input_grad_mapping = assert_unwrap(input_grad.mapping);

  // iterate over input_grad coords (the "many" side)
  return transform(
      input_grad_mapping.left_values(),
      [&](ParallelTensorSpaceCoordinate const &p) {
        // map input_grad coord to output_grad coord
        ParallelTensorSpaceCoordinate output_p = output_to_input_coord(p);
        MachineSpaceCoordinate dst_machine = input_grad_mapping.at_l(p);

        bidict<ParallelTensorSpaceCoordinate, MachineSpaceCoordinate>
            output_grad_mapping = assert_unwrap(output_grad.mapping);

        DynamicValueAttrs sharded_output_grad = output_grad;
        sharded_output_grad.mapping =
            bidict<ParallelTensorSpaceCoordinate, MachineSpaceCoordinate>{
                {output_p, output_grad_mapping.at_l(output_p)}};
        sharded_output_grad.shard_coord = output_p;

        DynamicValueAttrs sharded_output_fwd = output_fwd;
        sharded_output_fwd.mapping =
            bidict<ParallelTensorSpaceCoordinate, MachineSpaceCoordinate>{
                {output_p, output_grad_mapping.at_l(output_p)}};
        sharded_output_fwd.shard_coord = output_p;

        DynamicValueAttrs sharded_input_grad = input_grad;
        sharded_input_grad.mapping =
            bidict<ParallelTensorSpaceCoordinate, MachineSpaceCoordinate>{
                {p, dst_machine}};
        sharded_input_grad.shard_coord = p;

        DynamicNodeAttrs sharded_node = i.node_attrs;
        sharded_node.device_coord = dst_machine;

        return DynamicNodeInvocation{
            /*inputs=*/{
                {output_fwd_slot, sharded_output_fwd},
                {output_grad_slot, sharded_output_grad},
            },
            /*node_attrs=*/sharded_node,
            /*outputs=*/
            {
                {input_grad_slot, sharded_input_grad},
            },
        };
      });
}
static std::unordered_set<DynamicNodeInvocation>
    perform_shard_expansion_many_to_one(
        DynamicNodeInvocation const &i,
        std::function<ParallelTensorSpaceCoordinate(
            ParallelTensorSpaceCoordinate const &)> input_to_output_coord) {

  if (has_task_type(i.node_attrs, DynamicTaskType::FWD)) {
    auto const &[input_slot, input] = get_only(i.inputs);
    auto const &[output_slot, output] = get_only(i.outputs);

    bidict<ParallelTensorSpaceCoordinate, MachineSpaceCoordinate>
        input_mapping = assert_unwrap(input.mapping);
    bidict<ParallelTensorSpaceCoordinate, MachineSpaceCoordinate>
        output_mapping = assert_unwrap(output.mapping);

    return transform(input_mapping.left_values(),
                     [&](ParallelTensorSpaceCoordinate const &p) {
                       ParallelTensorSpaceCoordinate output_p =
                           input_to_output_coord(p);
                       MachineSpaceCoordinate dst_machine =
                           output_mapping.at_l(output_p);
                       return shard_invocation_for_binding(
                           i,
                           dst_machine,
                           OperatorAtomicTaskShardBinding{{
                               {input_slot.slot_name, p},
                               {output_slot.slot_name, output_p},
                           }});
                     });
  }

  // BWD case
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

  // group output_grad coords by their corresponding input_grad coord
  std::unordered_map<ParallelTensorSpaceCoordinate,
                     std::unordered_set<ParallelTensorSpaceCoordinate>>
      input_grad_to_output_grads;
  for (auto const &p : output_grad_mapping.left_values()) {
    input_grad_to_output_grads[input_to_output_coord(p)].insert(p);
  }

  std::unordered_set<DynamicNodeInvocation> result;
  for (auto const &[input_grad_p, output_grad_coords] :
       input_grad_to_output_grads) {

    MachineSpaceCoordinate dst_machine = input_grad_mapping.at_l(input_grad_p);

    // subset output_grad mapping to just this group's coords
    bidict<ParallelTensorSpaceCoordinate, MachineSpaceCoordinate>
        replica_mapping;
    for (auto const &p : output_grad_coords) {
      replica_mapping.equate(p, output_grad_mapping.at_l(p));
    }

    DynamicValueAttrs sharded_output_grad = output_grad;
    sharded_output_grad.mapping = replica_mapping;
    sharded_output_grad.shard_coord = input_grad_p;

    DynamicValueAttrs sharded_output_fwd = output_fwd;
    sharded_output_fwd.mapping = replica_mapping;
    sharded_output_fwd.shard_coord = input_grad_p;

    DynamicValueAttrs sharded_input_grad = input_grad;
    sharded_input_grad.mapping =
        bidict<ParallelTensorSpaceCoordinate, MachineSpaceCoordinate>{
            {input_grad_p, dst_machine}};
    sharded_input_grad.shard_coord = input_grad_p;

    DynamicNodeAttrs sharded_node = i.node_attrs;
    sharded_node.device_coord = dst_machine;

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

// Replicate/Reduction FWD — output has discard_copy=0..N-1, input always discard_copy=0
static std::unordered_set<DynamicNodeInvocation>
    perform_shard_expansion_for_replicate(DynamicNodeInvocation const &i) {
  return perform_shard_expansion_one_to_many(
      i, [](ParallelTensorSpaceCoordinate const &p) {
        return ParallelTensorSpaceCoordinate{
            p.sum_component, nonnegative_int{0}, p.shard_components};
      });
}

// Replicate BWD — many discard_copy inputs → one discard_copy=0 output
static std::unordered_set<DynamicNodeInvocation>
    perform_shard_expansion_for_replicate_bwd(DynamicNodeInvocation const &i) {
  return perform_shard_expansion_many_to_one(
      i, [](ParallelTensorSpaceCoordinate const &p) {
        return ParallelTensorSpaceCoordinate{
            p.sum_component, nonnegative_int{0}, p.shard_components};
      });
}

// Repartition FWD — output coord (high) → input coord (low)
static std::unordered_set<DynamicNodeInvocation>
    perform_shard_expansion_for_repartition(DynamicNodeInvocation const &i) {
  RepartitionAttrs attrs = i.node_attrs.op_attrs.value()
                               .get<PCGOperatorAttrs>()
                               .get<RepartitionAttrs>();
  relative_ff_dim_t rel_dim =
      relative_ff_dim_t_from_ff_dim_t(attrs.repartition_dim);
  nonnegative_int degree =
      attrs.repartition_degree.nonnegative_int_from_positive_int();

  return perform_shard_expansion_one_to_many(
      i, [=](ParallelTensorSpaceCoordinate const &p) {
        FFOrdered<nonnegative_int> input_shard = p.shard_components;
        input_shard.at(rel_dim) =
            p.shard_components.at(rel_dim) / degree; // ← /  not %
        return ParallelTensorSpaceCoordinate{
            p.sum_component, p.discard_copy_component, input_shard};
      });
}

// Repartition BWD — output_grad coord (high) → input_grad coord (low)
static std::unordered_set<DynamicNodeInvocation>
    perform_shard_expansion_for_repartition_bwd(
        DynamicNodeInvocation const &i) {
  RepartitionAttrs attrs = i.node_attrs.op_attrs.value()
                               .get<PCGOperatorAttrs>()
                               .get<RepartitionAttrs>();
  relative_ff_dim_t rel_dim =
      relative_ff_dim_t_from_ff_dim_t(attrs.repartition_dim);
  nonnegative_int degree =
      attrs.repartition_degree.nonnegative_int_from_positive_int();

  return perform_shard_expansion_many_to_one(
      i, [=](ParallelTensorSpaceCoordinate const &p) {
        FFOrdered<nonnegative_int> input_shard = p.shard_components;
        input_shard.at(rel_dim) =
            p.shard_components.at(rel_dim) / degree; // ← /  not %
        return ParallelTensorSpaceCoordinate{
            p.sum_component, p.discard_copy_component, input_shard};
      });
}

// Combine FWD — input coord (high) → output coord (low)
static std::unordered_set<DynamicNodeInvocation>
    perform_shard_expansion_for_combine(DynamicNodeInvocation const &i) {
  CombineAttrs attrs =
      i.node_attrs.op_attrs.value().get<PCGOperatorAttrs>().get<CombineAttrs>();
  relative_ff_dim_t rel_dim =
      relative_ff_dim_t_from_ff_dim_t(attrs.combine_dim);
  nonnegative_int degree =
      attrs.combine_degree.nonnegative_int_from_positive_int();

  return perform_shard_expansion_many_to_one(
      i, [=](ParallelTensorSpaceCoordinate const &p) {
        FFOrdered<nonnegative_int> output_shard = p.shard_components;
        output_shard.at(rel_dim) =
            p.shard_components.at(rel_dim) / degree; // ← correct
        return ParallelTensorSpaceCoordinate{
            p.sum_component, p.discard_copy_component, output_shard};
      });
}

// Combine BWD — input_grad coord (high) → output_grad coord (low)
static std::unordered_set<DynamicNodeInvocation>
    perform_shard_expansion_for_combine_bwd(DynamicNodeInvocation const &i) {
  CombineAttrs attrs =
      i.node_attrs.op_attrs.value().get<PCGOperatorAttrs>().get<CombineAttrs>();
  relative_ff_dim_t rel_dim =
      relative_ff_dim_t_from_ff_dim_t(attrs.combine_dim);
  nonnegative_int degree =
      attrs.combine_degree.nonnegative_int_from_positive_int();

  return perform_shard_expansion_one_to_many(
      i, [=](ParallelTensorSpaceCoordinate const &p) {
        FFOrdered<nonnegative_int> output_shard = p.shard_components;
        output_shard.at(rel_dim) =
            p.shard_components.at(rel_dim) / degree; // ← / not %
        return ParallelTensorSpaceCoordinate{
            p.sum_component, p.discard_copy_component, output_shard};
      });
}

// Reduction FWD — input coord (sum=0..N-1) → output coord (sum=0)
static std::unordered_set<DynamicNodeInvocation>
    perform_shard_expansion_for_reduction(DynamicNodeInvocation const &i) {
  return perform_shard_expansion_many_to_one(
      i, [](ParallelTensorSpaceCoordinate const &p) {
        return ParallelTensorSpaceCoordinate{
            nonnegative_int{0}, // ← output always has sum=0
            p.discard_copy_component,
            p.shard_components};
      });
}

// Reduction BWD — output_grad coord (sum=0) → input_grad coord (sum=0..N-1)
static std::unordered_set<DynamicNodeInvocation>
    perform_shard_expansion_for_reduction_bwd(DynamicNodeInvocation const &i) {
  return perform_shard_expansion_many_to_one(
      i, [](ParallelTensorSpaceCoordinate const &p) {
        return ParallelTensorSpaceCoordinate{
            p.sum_component, nonnegative_int{0}, p.shard_components};
      });
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

static std::unordered_set<DynamicNodeInvocation>
    perform_shard_expansion_for_parallel_op(DynamicNodeInvocation const &i) {
  ASSERT(is_parallel_op_attrs(i.node_attrs));

  PCGOperatorAttrs const pcg =
      i.node_attrs.op_attrs.value().get<PCGOperatorAttrs>();

  // forward dispatch
  if (has_task_type(i.node_attrs, DynamicTaskType::FWD)) {
    if (pcg.has<ReplicateAttrs>()) {
      return perform_shard_expansion_for_replicate(i);
    }
    if (pcg.has<RepartitionAttrs>()) {
      return perform_shard_expansion_for_repartition(i);
    }
    if (pcg.has<CombineAttrs>()) {
      return perform_shard_expansion_for_combine(i);
    }
    if (pcg.has<ReductionAttrs>()) {
      return perform_shard_expansion_for_reduction(i);
    }
  }

  // backward dispatch
  if (has_task_type(i.node_attrs, DynamicTaskType::BWD)) {
    if (pcg.has<ReplicateAttrs>()) {
      return perform_shard_expansion_for_replicate_bwd(i);
    }
    if (pcg.has<RepartitionAttrs>()) {
      return perform_shard_expansion_for_repartition_bwd(i);
    }
    if (pcg.has<CombineAttrs>()) {
      return perform_shard_expansion_for_combine_bwd(i);
    }
    if (pcg.has<ReductionAttrs>()) {
      return perform_shard_expansion_for_reduction_bwd(i);
    }
  }
  PANIC("unhandled parallel op task_type: {}", i.node_attrs.task_type);
}

std::unordered_set<DynamicNodeInvocation>
    perform_shard_expansion_for_invocation(DynamicNodeInvocation const &i) {
  if (i.node_attrs.op_attrs.has_value() &&
      i.node_attrs.op_attrs.value().is_copy()) {
    return perform_shard_expansion_for_copy(i);
  }

  if (is_parallel_op_attrs(i.node_attrs)) {
    return perform_shard_expansion_for_parallel_op(i);
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
