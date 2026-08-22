#include "task-spec/dynamic_graph/pass_expansion.h"
#include "task-spec/dynamic_graph/dynamic_node_invocation.h"
#include "task-spec/dynamic_graph/dynamic_open_dataflow_graph.h"
#include "task-spec/dynamic_graph/dynamic_tensor_role.h"
#include "task-spec/dynamic_graph/training_operation_attrs.h"
#include "utils/containers/are_all_same.h"
#include "utils/containers/concat_vectors.h"
#include "utils/containers/contains_duplicates.h"
#include "utils/containers/filter.h"
#include "utils/containers/flatmap.h"
#include "utils/containers/get_only.h"
#include "utils/containers/map_from_pairs.h"
#include "utils/containers/map_values.h"
#include "utils/containers/map_values2.h"
#include "utils/containers/merge_disjoint_maps.h"
#include "utils/containers/multiset_of.h"
#include "utils/containers/range.h"
#include "utils/containers/repeat_until_converged.h"
#include "utils/containers/set_of.h"
#include "utils/containers/transform.h"
#include "utils/containers/zip_with.h"

namespace FlexFlow {

void require_node_might_be_pass_expanded(DynamicNodeAttrs const &n) {
  if (assert_unwrap(n.op_attrs).is_copy()) {
    return;
  }

  ASSERT(n.task_type.has_value(), n);
}

void require_node_might_not_be_pass_expanded(DynamicNodeAttrs const &n) {
  if (assert_unwrap(n.op_attrs).is_copy()) {
    return;
  }

  ASSERT(!n.task_type.has_value(), n);
  ASSERT(!assert_unwrap(n.op_attrs).is_gradient_reduction());
}

void require_slot_is_not_pass_expanded(DynamicTensorSlot const &s) {
  ASSERT(!s.slot_tensor_role.has_value(), s);
}

void require_value_is_pass_expanded(DynamicValueAttrs const &v) {
  ASSERT(v.role.has_value(), v);
}

void require_value_is_not_pass_expanded(DynamicValueAttrs const &v) {
  ASSERT(!v.role.has_value(), v);
  ASSERT(!v.subgradient_id.has_value(), v);
}

void require_invocation_is_fully_pass_expanded(
    DynamicNodeInvocation const &invocation) {
  auto require_slot_is_pass_expanded = [&](DynamicTensorSlot const &s) {
    if (dynamic_node_invocation_get_op_type(invocation) ==
        TrainingOpType{TrainingOnlyOpType::COPY}) {
      return;
    }

    ASSERT(s.slot_tensor_role.has_value(), s);
  };

  require_invocation_fully_satisfies(invocation,
                                     require_node_might_be_pass_expanded,
                                     require_value_is_pass_expanded,
                                     require_slot_is_pass_expanded);
}

void require_invocation_is_ready_for_pass_expansion(
    DynamicNodeInvocation const &invocation) {
  require_invocation_fully_satisfies(invocation,
                                     require_node_might_not_be_pass_expanded,
                                     require_value_is_not_pass_expanded,
                                     require_slot_is_not_pass_expanded);
}

void require_graph_is_fully_pass_expanded(DynamicOpenDataflowGraph const &g) {
  require_full_dynamic_graph_satisfies(
      g, require_invocation_is_fully_pass_expanded);
}

void require_graph_is_ready_for_pass_expansion(
    DynamicOpenDataflowGraph const &g) {
  require_full_dynamic_graph_satisfies(
      g, require_invocation_is_ready_for_pass_expansion);
}

std::set<DynamicValueAttrs>
    determine_intermediate_values_needed_to_compute_gradients_of_value(
        DynamicOpenDataflowGraph const &g, DynamicValueAttrs const &val) {
  auto get_values_immediately_needed_to_compute_gradients_of_needed =
      [&](std::set<DynamicValueAttrs> const &needed)
      -> std::set<DynamicValueAttrs> {
    std::set<DynamicValueAttrs> additional = flatmap(
        needed, [&](DynamicValueAttrs const &v) -> std::set<DynamicValueAttrs> {
          std::set<InternalDynamicSlotSite> sinks =
              dynamic_graph_find_sinks_of_value(g, v);

          return flatmap(sinks,
                         [&](InternalDynamicSlotSite const &sink)
                             -> std::set<DynamicValueAttrs> {
                           DynamicNodeInvocation sink_invocation =
                               dynamic_graph_get_invocation_for_id(
                                   g, sink.invocation_id);

                           return set_of(values(sink_invocation.outputs));
                         });
        });

    return set_union(needed, additional);
  };

  std::set<DynamicValueAttrs> result = repeat_until_converged(
      std::set<DynamicValueAttrs>{val},
      get_values_immediately_needed_to_compute_gradients_of_needed);

  ASSERT(contains(result, val));
  return result;
}

std::set<DynamicValueAttrs>
    determine_intermediate_values_needed_for_gradient_computation(
        DynamicOpenDataflowGraph const &g) {
  auto value_is_fundamentally_required =
      [&](DynamicValueAttrs const &v) -> bool {
    DynamicSlotSite source = dynamic_graph_find_source_of_value(g, v);

    if (source.is_external()) {
      return true;
    }

    InternalDynamicSlotSite internal_source = source.require_internal();
    ASSERT(internal_source.direction == TensorDirection::OUTPUT);

    DynamicNodeInvocation source_invocation =
        dynamic_graph_get_invocation_for_id(g, internal_source.invocation_id);

    TrainingOpType op_type =
        dynamic_node_invocation_get_op_type(source_invocation);

    TrainingOpType weight_op_type = TrainingOpType{OperatorType::WEIGHT};
    TrainingOpType input_op_type = TrainingOpType{OperatorType::INPUT};

    if (op_type == weight_op_type) {
      return true;
    } else if (op_type == input_op_type) {
      return assert_unwrap(v.create_grad);
    } else {
      return false;
    }
  };

  std::set<DynamicValueAttrs> fundamentally_required_values =
      filter(set_of(get_dynamic_values(g)), value_is_fundamentally_required);

  std::set<DynamicValueAttrs> required_values = flatmap(
      fundamentally_required_values,
      [&](DynamicValueAttrs const &fundamentally_required_value)
          -> std::set<DynamicValueAttrs> {
        return determine_intermediate_values_needed_to_compute_gradients_of_value(
            g, fundamentally_required_value);
      });

  return required_values;
}

std::set<dynamic_invocation_id_t>
    determine_invocations_needed_in_backward_pass_for_gradient_computation(
        DynamicOpenDataflowGraph const &g) {
  std::set<DynamicValueAttrs> required_values =
      determine_intermediate_values_needed_for_gradient_computation(g);

  auto get_sink_invocations_for_value =
      [&](DynamicValueAttrs const &v) -> std::set<dynamic_invocation_id_t> {
    return transform(dynamic_graph_find_sinks_of_value(g, v),
                     [&](InternalDynamicSlotSite const &sink_site)
                         -> dynamic_invocation_id_t {
                       ASSERT(sink_site.direction == TensorDirection::INCOMING);
                       return sink_site.invocation_id;
                     });
  };

  return flatmap(required_values, get_sink_invocations_for_value);
}

DynamicTensorSlot pass_expand_slot(DynamicTensorSlot const &s,
                                   FwbTensorType tensor_type) {
  require_slot_is_not_pass_expanded(s);

  DynamicTensorSlot result = s;
  result.slot_tensor_role =
      dynamic_tensor_role_from_fwb_tensor_type(tensor_type);
  return result;
}

DynamicValueAttrs pass_expand_value(DynamicValueAttrs const &v,
                                    FwbTensorType tensor_type) {
  require_value_is_not_pass_expanded(v);

  DynamicValueAttrs result = v;
  result.role = DynamicTensorRole{tensor_type};
  return result;
};

DynamicNodeAttrs pass_expand_node(DynamicNodeAttrs const &n,
                                  DynamicTaskType task_type) {
  require_node_might_not_be_pass_expanded(n);

  ASSERT(task_type == DynamicTaskType::FWD ||
         task_type == DynamicTaskType::BWD);

  {
    TrainingOperationAttrs op_attrs = assert_unwrap(n.op_attrs);

    if (op_attrs.is_copy()) {
      return n;
    }
  }

  DynamicNodeAttrs result = n;
  result.task_type = task_type;
  return result;
}

DynamicNodeInvocation perform_fwd_pass_expansion_for_invocation(
    DynamicNodeInvocation const &invocation) {

  require_invocation_is_ready_for_pass_expansion(invocation);

  TrainingOperationAttrs op_attrs =
      assert_unwrap(invocation.node_attrs.op_attrs);

  auto to_fwd_value = [](DynamicValueAttrs const &v) -> DynamicValueAttrs {
    return pass_expand_value(v, FwbTensorType::FORWARD);
  };

  auto to_fwd = [](DynamicTensorSlot const &k, DynamicValueAttrs const &v) {
    return std::pair{
        pass_expand_slot(k, FwbTensorType::FORWARD),
        pass_expand_value(v, FwbTensorType::FORWARD),
    };
  };

  DynamicNodeInvocation result = [&]() -> DynamicNodeInvocation {
    if (op_attrs.is_copy()) {
      return DynamicNodeInvocation{
          /*inputs=*/map_values(invocation.inputs, to_fwd_value),
          /*node_attrs=*/invocation.node_attrs,
          /*outputs=*/map_values(invocation.outputs, to_fwd_value),
      };
    } else {
      return DynamicNodeInvocation{
          /*inputs=*/
          transform(invocation.inputs, to_fwd),
          /*node_attrs=*/
          pass_expand_node(invocation.node_attrs, DynamicTaskType::FWD),
          /*outputs=*/
          transform(invocation.outputs, to_fwd),
      };
    }
  }();

  require_invocation_is_fully_pass_expanded(result);

  return result;
}

DynamicNodeInvocation perform_bwd_pass_expansion_for_invocation(
    DynamicNodeInvocation const &invocation) {

  require_invocation_is_ready_for_pass_expansion(invocation);

  TrainingOperationAttrs op_attrs =
      assert_unwrap(invocation.node_attrs.op_attrs);

  auto to_fwd = [](DynamicTensorSlot const &k, DynamicValueAttrs const &v) {
    return std::pair{
        pass_expand_slot(k, FwbTensorType::FORWARD),
        pass_expand_value(v, FwbTensorType::FORWARD),
    };
  };

  auto to_grad_value = [](DynamicValueAttrs const &v) {
    return pass_expand_value(v, FwbTensorType::GRADIENT);
  };

  auto to_grad = [](DynamicTensorSlot const &k, DynamicValueAttrs const &v) {
    return std::pair{
        pass_expand_slot(k, FwbTensorType::GRADIENT),
        pass_expand_value(v, FwbTensorType::GRADIENT),
    };
  };

  DynamicNodeInvocation result = [&]() -> DynamicNodeInvocation {
    if (op_attrs.is_copy()) {
      return DynamicNodeInvocation{
          /*inputs=*/map_values(invocation.outputs, to_grad_value),
          /*node_attrs=*/invocation.node_attrs,
          /*outputs=*/map_values(invocation.inputs, to_grad_value),
      };
    } else if (training_op_attrs_has_op_type(op_attrs,
                                             OperatorType::REPLICATE)) {
      return DynamicNodeInvocation{
          /*inputs=*/{
              transform(invocation.outputs, to_grad),
          },
          /*node_attrs=*/
          pass_expand_node(invocation.node_attrs, DynamicTaskType::BWD),
          /*outputs=*/
          {
              transform(invocation.inputs, to_grad),
          },
      };
    } else {
      return DynamicNodeInvocation{
          /*inputs=*/
          merge_disjoint_maps(std::vector{
              transform(invocation.inputs, to_fwd),
              transform(invocation.outputs, to_fwd),
              transform(invocation.outputs, to_grad),
          }),
          /*node_attrs=*/
          pass_expand_node(invocation.node_attrs, DynamicTaskType::BWD),
          /*outputs=*/
          transform(invocation.inputs, to_grad),
      };
    };
  }();

  require_invocation_is_fully_pass_expanded(result);

  return result;
}

static std::map<int, subgradient_id_t>
    choose_subgradient_ids(std::set<int> const &invocations) {
  const std::vector<TensorSlotName> slot_names = {
      TensorSlotName::INPUT_00,
      TensorSlotName::INPUT_01,
      TensorSlotName::INPUT_02,
      TensorSlotName::INPUT_03,
      TensorSlotName::INPUT_04,
      TensorSlotName::INPUT_05,
      TensorSlotName::INPUT_06,
      TensorSlotName::INPUT_07,
      TensorSlotName::INPUT_08,
      TensorSlotName::INPUT_09,
      TensorSlotName::INPUT_10,
      TensorSlotName::INPUT_11,
      TensorSlotName::INPUT_12,
      TensorSlotName::INPUT_13,
      TensorSlotName::INPUT_14,
      TensorSlotName::INPUT_15,
  };
  std::map<int, subgradient_id_t> result = map_from_pairs(
      zip_with(vector_of(invocations),
               slot_names,
               [](int idx, TensorSlotName slot_name) {
                 return std::pair{idx, subgradient_id_t{slot_name}};
               }));
  ASSERT(result.size() == invocations.size());
  return result;
}

static std::vector<DynamicNodeInvocation> reduce_gradients_for_invocations(
    std::vector<DynamicNodeInvocation> const &invocations) {
  std::multiset<DynamicValueAttrs> outputs =
      multiset_of(flatmap(invocations, [](DynamicNodeInvocation const &i) {
        return vector_of(values(i.outputs));
      }));
  std::set<DynamicValueAttrs> unique_outputs = set_of(outputs);

  std::map<DynamicNodeInvocation, int> invocation_index =
      map_from_keys_and_values(invocations, range(invocations.size()));

  std::map<DynamicValueAttrs, std::set<int>> output_invocation_set;
  for (int idx : range(invocations.size())) {
    DynamicNodeInvocation const &i = invocations[idx];
    for (DynamicValueAttrs const &v : values(i.outputs)) {
      if (outputs.count(v) > 1) {
        output_invocation_set[v].insert(idx);
      }
    }
  }
  std::map<DynamicValueAttrs, std::map<int, subgradient_id_t>>
      output_subgradient_ids =
          map_values(output_invocation_set, choose_subgradient_ids);

  std::map<DynamicValueAttrs, DynamicNodeInvocation> gradient_reductions =
      map_values2(output_invocation_set,
                  [&](DynamicValueAttrs const &output,
                      std::set<int> const &invocation_set) {
                    return DynamicNodeInvocation{
                        /*inputs=*/map_from_pairs(transform(
                            values(output_subgradient_ids.at(output)),
                            [&](subgradient_id_t id) {
                              DynamicTensorSlot slot{
                                  /*slot_name=*/id.gradient_reduction_slot,
                                  /*slot_tensor_role=*/
                                  DynamicTensorRole{FwbTensorType::GRADIENT},
                                  /*task_shard=*/std::nullopt,
                              };

                              DynamicValueAttrs input = output;
                              input.subgradient_id = id;

                              return std::pair{slot, input};
                            })),
                        /*node_attrs=*/
                        DynamicNodeAttrs{
                            /*task_type=*/DynamicTaskType::BWD,
                            /*device_ids=*/std::nullopt,
                            /*mapping=*/std::nullopt,
                            /*op_attrs=*/
                            TrainingOperationAttrs{GradientReductionAttrs{}},
                            /*layer_guid=*/
                            dynamic_layer_guid_t{
                                dynamic_gradient_reduction_layer_guid_t{}},
                            /*per_device_op_state=*/std::nullopt,
                        },
                        /*outputs=*/
                        std::map<DynamicTensorSlot, DynamicValueAttrs>{
                            {
                                DynamicTensorSlot{
                                    /*slot_name=*/TensorSlotName::OUTPUT,
                                    /*slot_tensor_role=*/
                                    DynamicTensorRole{FwbTensorType::GRADIENT},
                                    /*task_shard=*/std::nullopt,
                                },
                                output,
                            },
                        },
                    };
                  });

  return concat_vectors(
      transform(range(invocations.size()),
                [&](int idx) {
                  DynamicNodeInvocation mapped_invocation = invocations.at(idx);
                  mapped_invocation.outputs = map_values(
                      mapped_invocation.outputs,
                      [&](DynamicValueAttrs const &output) {
                        DynamicValueAttrs mapped_output = output;
                        if (output_subgradient_ids.count(output)) {
                          mapped_output.subgradient_id =
                              output_subgradient_ids.at(output).at(idx);
                        }
                        return mapped_output;
                      });
                  return mapped_invocation;
                }),
      vector_of(values(gradient_reductions)));
}

// Like flatmap_dynamic_invocation_set except we replace the creation of
// duplicate values with GradientReduction
static DynamicOpenDataflowGraph
    flatmap_dynamic_invocation_set_with_gradient_reduction(
        DynamicOpenDataflowGraph const &g,
        std::function<std::set<DynamicNodeInvocation>(
            DynamicNodeInvocation const &)> const &f) {
  std::set<DynamicNodeInvocation> current_invocation_set =
      get_dynamic_invocation_set(g);
  std::vector<DynamicNodeInvocation> new_invocation_set =
      flatmap(vector_of(current_invocation_set), f);

  ASSERT(!contains_duplicates(new_invocation_set));

  new_invocation_set = reduce_gradients_for_invocations(new_invocation_set);

  return dynamic_open_dataflow_graph_from_invocation_set(
      set_of(new_invocation_set));
}

DynamicOpenDataflowGraph
    perform_pass_expansion(DynamicOpenDataflowGraph const &g) {

  require_graph_is_ready_for_pass_expansion(g);

  std::set<dynamic_invocation_id_t> needed_in_bwd_pass =
      determine_invocations_needed_in_backward_pass_for_gradient_computation(g);

  DynamicOpenDataflowGraph result =
      flatmap_dynamic_invocation_set_with_gradient_reduction(
          g, [&](DynamicNodeInvocation const &invocation) {
            dynamic_invocation_id_t invocation_id =
                dynamic_graph_get_id_for_invocation(g, invocation);

            if (contains(needed_in_bwd_pass, invocation_id)) {
              return std::set{
                  perform_fwd_pass_expansion_for_invocation(invocation),
                  perform_bwd_pass_expansion_for_invocation(invocation),
              };
            } else {
              return std::set{
                  perform_fwd_pass_expansion_for_invocation(invocation),
              };
            }
          });

  require_graph_is_fully_pass_expanded(result);

  return result;
}

} // namespace FlexFlow
