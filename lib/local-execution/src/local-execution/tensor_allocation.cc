#include "local-execution/tensor_allocation.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "task-spec/dynamic_graph/dynamic_open_dataflow_graph.h"
#include "task-spec/dynamic_graph/dynamic_tensor_accessor.dtg.h"
#include "utils/bidict/generate_bidict.h"
#include "utils/containers/all_are_true.h"
#include "utils/containers/contains_key.h"
#include "utils/containers/map_values.h"
#include "utils/containers/unordered_set_of.h"
#include "utils/optional.h"

namespace FlexFlow {

bool no_tensors_are_allocated(DynamicOpenDataflowGraph const &g) {
  return all_are_true(
      transform(get_dynamic_values(g), [](DynamicValueAttrs const &v) -> bool {
        return !v.accessor.has_value();
      }));
}

bool all_tensors_are_allocated(DynamicOpenDataflowGraph const &g) {
  return all_are_true(
      transform(get_dynamic_values(g), [](DynamicValueAttrs const &v) -> bool {
        return v.accessor.has_value();
      }));
}

bool tensors_are_ready_for_allocation(DynamicOpenDataflowGraph const &g) {
  return all_are_true(
      transform(get_dynamic_values(g), [](DynamicValueAttrs const &v) -> bool {
        return v.parallel_tensor_shape.has_value();
      }));
}

DynamicValueAttrs
    perform_tensor_allocation_for_value(DynamicValueAttrs const &value,
                                        Allocator &allocator) {
  ASSERT(value.accessor == std::nullopt);

  TensorShape shape =
      get_piece_shape(assert_unwrap(value.parallel_tensor_shape));

  GenericTensorAccessorW accessor = allocator.allocate_tensor(shape);

  DynamicValueAttrs result = value;
  result.accessor = DynamicTensorAccessor{accessor};

  return result;
}

DynamicOpenDataflowGraph perform_tensor_allocation(
    DynamicOpenDataflowGraph const &g,
    std::unordered_map<DynamicValueAttrs, DynamicTensorAccessor> const
        &preallocated,
    Allocator &allocator) {
  ASSERT(no_tensors_are_allocated(g));
  ASSERT(tensors_are_ready_for_allocation(g));
  for (DynamicValueAttrs const &v : keys(preallocated)) {
    ASSERT(v.accessor == std::nullopt);
  }

  std::unordered_set<DynamicValueAttrs> all_values =
      unordered_set_of(get_dynamic_values(g));

  bidict<DynamicValueAttrs, DynamicValueAttrs> unallocated_to_allocated =
      generate_bidict(
          all_values, [&](DynamicValueAttrs const &v) -> DynamicValueAttrs {
            if (contains_key(preallocated, v)) {
              DynamicValueAttrs result = v;
              result.accessor = preallocated.at(v);
              return result;
            } else {
              return perform_tensor_allocation_for_value(v, allocator);
            }
          });

  DynamicOpenDataflowGraph result = transform_dynamic_invocation_set(
      g, [&](DynamicNodeInvocation const &i) -> DynamicNodeInvocation {
        return DynamicNodeInvocation{
            /*inputs=*/map_values(
                i.inputs,
                [&](DynamicValueAttrs const &v) -> DynamicValueAttrs {
                  return unallocated_to_allocated.at_l(v);
                }),
            /*node_attrs=*/i.node_attrs,
            /*outputs=*/
            map_values(i.outputs,
                       [&](DynamicValueAttrs const &v) -> DynamicValueAttrs {
                         return unallocated_to_allocated.at_l(v);
                       }),
        };
      });

  ASSERT(all_tensors_are_allocated(result));

  return result;
}

} // namespace FlexFlow
