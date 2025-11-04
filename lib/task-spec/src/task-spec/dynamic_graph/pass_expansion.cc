#include "task-spec/dynamic_graph/pass_expansion.h"
#include "task-spec/dynamic_graph/dynamic_open_dataflow_graph.h"
#include "utils/containers/concat_vectors.h"

namespace FlexFlow {

bool node_is_pass_expanded(DynamicNodeAttrs const &n) {
  return n.pass_type.has_value();
}

bool value_is_pass_expanded(DynamicValueAttrs const &v) {
  return v.tensor_type.has_value();
}

bool no_part_of_graph_is_pass_expanded(DynamicOpenDataflowGraph const &g) {
  return no_part_of_dynamic_graph_satisfies(
    g, node_is_pass_expanded, value_is_pass_expanded);
}

bool graph_is_fully_pass_expanded(DynamicOpenDataflowGraph const &g) {
  return full_dynamic_graph_satisfies(
    g, node_is_pass_expanded, value_is_pass_expanded);
}

DynamicNodeInvocation 
  perform_pass_expansion_for_task(DynamicNodeInvocation const &task, PassType pass) {

  switch (pass) {
    case PassType::FWD:
      return perform_fwd_pass_expansion_for_task(task);
    case PassType::BWD:
      return perform_bwd_pass_expansion_for_task(task);
    default:
      PANIC("Unhandled Pass", pass);
  }
}

DynamicValueAttrs pass_expand_value(DynamicValueAttrs const &v, FwbTensorType tensor_type){
  ASSERT(!value_is_pass_expanded(v));

  DynamicValueAttrs result = v;
  result.tensor_type = tensor_type;
  return result;  
};

DynamicNodeAttrs pass_expand_node(DynamicNodeAttrs const &n, PassType pass_type) {
  ASSERT(!node_is_pass_expanded(n));

  DynamicNodeAttrs result = n;
  result.pass_type = pass_type;
  return result;
}

DynamicNodeInvocation 
  perform_fwd_pass_expansion_for_task(DynamicNodeInvocation const &task) {
  
  auto get_fwd_tensor = 
                [](DynamicValueAttrs const &v) {
                  return pass_expand_value(v, FwbTensorType::FORWARD);
                };

  return DynamicNodeInvocation{
    /*inputs=*/
      transform(task.inputs, get_fwd_tensor),
    /*node_attrs=*/
      pass_expand_node(task.node_attrs, PassType::FWD),
    /*outputs=*/
      transform(task.inputs, get_fwd_tensor),
  };
}

DynamicNodeInvocation 
  perform_bwd_pass_expansion_for_task(DynamicNodeInvocation const &invocation) {

  auto get_fwd_tensor = 
                [](DynamicValueAttrs const &v) {
                  return pass_expand_value(v, FwbTensorType::FORWARD);
                };

  auto get_bwd_tensor = 
                [](DynamicValueAttrs const &v) {
                  return pass_expand_value(v, FwbTensorType::GRADIENT);
                };

  return DynamicNodeInvocation{
    /*inputs=*/
      concat_vectors(
        transform(invocation.inputs, get_fwd_tensor),
        transform(invocation.outputs, get_bwd_tensor)),
    /*node_attrs=*/
      pass_expand_node(invocation.node_attrs, PassType::BWD),
    /*outputs=*/
      transform(invocation.inputs, get_bwd_tensor),
  };
}


DynamicOpenDataflowGraph
  perform_pass_expansion(DynamicOpenDataflowGraph const &g) {

  ASSERT(no_part_of_graph_is_pass_expanded(g));

  auto pass_expand_node = [](DynamicNodeAttrs const &n, PassType pass_type) -> DynamicNodeAttrs {
    ASSERT(!node_is_pass_expanded(n));

    DynamicNodeAttrs result = n;
    result.pass_type = pass_type;
    return result;
  };

  // return transform_dynamic_task_set(
  //   g,
  //   [](
  //
  // for (DynamicTask const &t: get_dynamic_task_set(g)) {
  //
  // }
}


} // namespace FlexFlow
