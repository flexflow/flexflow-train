#include "realm-execution/instance_allocation.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "op-attrs/tensor_shape.dtg.h"
#include "realm-execution/realm_context.h"
#include "task-spec/dynamic_graph/dynamic_open_dataflow_graph.h"
#include "task-spec/dynamic_graph/dynamic_tensor_accessor.dtg.h"
#include "utils/bidict/generate_bidict.h"
#include "utils/containers/all_are_true.h"
#include "utils/containers/contains_key.h"
#include "utils/containers/make.h"
#include "utils/containers/map_values.h"
#include "utils/containers/unordered_set_of.h"
#include "utils/exception.h"
#include "utils/optional.h"

namespace FlexFlow {

bool no_instances_are_allocated(DynamicOpenDataflowGraph const &g) {
  return all_are_true(
      transform(get_dynamic_values(g), [](DynamicValueAttrs const &v) -> bool {
        return !v.accessor.has_value() && !v.instance.has_value();
      }));
}

bool all_instances_are_allocated(DynamicOpenDataflowGraph const &g) {
  return all_are_true(
      transform(get_dynamic_values(g), [](DynamicValueAttrs const &v) -> bool {
        return v.instance.has_value();
      }));
}

bool instances_are_ready_for_allocation(DynamicOpenDataflowGraph const &g) {
  return all_are_true(
      transform(get_dynamic_values(g), [](DynamicValueAttrs const &v) -> bool {
        return v.parallel_tensor_shape.has_value();
      }));
}

DynamicValueAttrs
    perform_instance_allocation_for_value(DynamicValueAttrs const &value,
                                          RealmContext &ctx) {
  ASSERT(value.accessor == std::nullopt);
  ASSERT(value.instance == std::nullopt);

  TensorShape shape = get_piece_shape(value.parallel_tensor_shape.value());

  Realm::Memory memory = Realm::Memory::NO_MEMORY; // FIXME
  auto [instance, ready] =
      ctx.create_instance(memory, shape, Realm::ProfilingRequestSet());

  DynamicValueAttrs result = value;
  result.instance = instance;

  return result;
}

std::pair<DynamicOpenDataflowGraph, Realm::Event> perform_instance_allocation(
    DynamicOpenDataflowGraph const &g,
    std::unordered_map<DynamicValueAttrs, DynamicTensorAccessor> const
        &preallocated,
    RealmContext &ctx) {
  ASSERT(no_instances_are_allocated(g));
  ASSERT(instances_are_ready_for_allocation(g));
  for (DynamicValueAttrs const &v : keys(preallocated)) {
    ASSERT(v.accessor == std::nullopt);
    ASSERT(v.instance == std::nullopt);
  }

  std::unordered_set<DynamicValueAttrs> all_values =
      unordered_set_of(get_dynamic_values(g));

  bidict<DynamicValueAttrs, DynamicValueAttrs> unallocated_to_allocated =
      generate_bidict(all_values,
                      [&](DynamicValueAttrs const &v) -> DynamicValueAttrs {
                        if (contains_key(preallocated, v)) {
                          // FIXME: Attach external instance to existing
                          // allocation and use that
                          NOT_IMPLEMENTED();
                        } else {
                          return perform_instance_allocation_for_value(v, ctx);
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

  ASSERT(all_instances_are_allocated(result));

  return std::pair{result, ctx.get_outstanding_events()};
}

} // namespace FlexFlow
