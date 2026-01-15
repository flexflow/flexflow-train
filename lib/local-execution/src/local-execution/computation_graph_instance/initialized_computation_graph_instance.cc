#include "local-execution/computation_graph_instance/initialized_computation_graph_instance.h"
#include "kernels/allocation.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "op-attrs/tensor_shape.dtg.h"
#include "task-spec/dynamic_graph/dynamic_open_dataflow_graph.h"
#include "utils/containers/transform.h"
#include "utils/exception.h"
#include "utils/optional.h"
#include <optional>
#include <vector>

namespace FlexFlow {

bool node_true(DynamicNodeAttrs const &n) {
  return true;
}

bool node_false(DynamicNodeAttrs const &n) {
  return true;
}

bool slot_true(DynamicTensorSlot const &s) {
  return true;
}

bool slot_false(DynamicTensorSlot const &s) {
  return true;
}

bool value_is_allocated(DynamicValueAttrs const &v) {
  return v.accessor.has_value();
}

bool no_part_of_graph_is_allocated(DynamicOpenDataflowGraph const &g) {
  return no_part_of_dynamic_graph_satisfies(
      g, node_false, value_is_allocated, slot_false);
}

bool graph_is_fully_allocated(DynamicOpenDataflowGraph const &g) {
  return full_dynamic_graph_satisfies(
      g, node_true, value_is_allocated, slot_true);
}
InitializedComputationGraphInstance::InitializedComputationGraphInstance(
    DynamicOpenDataflowGraph dg, Allocator &alloc)
    : initialized_dataflow_graph(dg), allocator(alloc) {}

DynamicValueAttrs allocate_value(
    DynamicValueAttrs const &v,
    bidict<dynamic_tensor_guid_t, GenericTensorAccessorW> &allocated_tensors,
    Allocator &allocator) {
  DynamicValueAttrs result = v;
  if (allocated_tensors.contains_l(result.tensor_guid)) {
    result.accessor = allocated_tensors.at_l(result.tensor_guid);
  } else {
    TensorShape shape =
        get_piece_shape(assert_unwrap(result.parallel_tensor_shape));
    GenericTensorAccessorW accessor = allocator.allocate_tensor(shape);
    allocated_tensors.equate(result.tensor_guid, accessor);
    result.accessor = accessor;
  }
  return result;
};

InitializedComputationGraphInstance initialize_computation_graph_instance(
    ComputationGraphInstance const &instance,
    bidict<dynamic_tensor_guid_t,
           std::variant<GenericTensorAccessorW, GenericTensorAccessorR>> const
        &input_tensors,
    Allocator &allocator) {
  // FIXME: initialize from input_tensors, may require modifying
  // dynamic_value_attrs to permit R tensors
  bidict<dynamic_tensor_guid_t, GenericTensorAccessorW> allocated_tensors;

  DynamicOpenDataflowGraph const &g = instance.expanded_dataflow_graph;
  ASSERT(no_part_of_graph_is_allocated(g));

  DynamicOpenDataflowGraph result = transform_dynamic_invocation_set(
      g, [&](DynamicNodeInvocation const &invocation) {
        auto allocate = [&](DynamicTensorSlot const &k,
                            DynamicValueAttrs const &v) {
          return std::pair{
              k,
              allocate_value(v, allocated_tensors, allocator),
          };
        };
        return DynamicNodeInvocation{
            /*inputs=*/
            transform(invocation.inputs, allocate),
            /*node_attrs=*/
            invocation.node_attrs,
            /*outputs=*/
            transform(invocation.outputs, allocate),
        };
      });

  ASSERT(graph_is_fully_allocated(result));

  return InitializedComputationGraphInstance{result, allocator};
}

std::unordered_map<layer_guid_t, std::optional<milliseconds_t>>
    perform_forward_pass_for_computation_graph_instance(
        InitializedComputationGraphInstance const &instance) {

  NOT_IMPLEMENTED();
}

std::unordered_map<layer_guid_t, std::optional<milliseconds_t>>
    perform_backward_pass_for_computation_graph_instance(
        InitializedComputationGraphInstance const &instance) {

  NOT_IMPLEMENTED();
}

} // namespace FlexFlow
