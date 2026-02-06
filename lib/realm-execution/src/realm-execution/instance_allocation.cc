#include "realm-execution/instance_allocation.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "op-attrs/tensor_shape.dtg.h"
#include "realm-execution/realm_context.h"
#include "task-spec/dynamic_graph/dynamic_node_attrs.dtg.h"
#include "task-spec/dynamic_graph/dynamic_open_dataflow_graph.h"
#include "task-spec/dynamic_graph/dynamic_tensor_accessor.dtg.h"
#include "task-spec/dynamic_graph/dynamic_value_attrs.dtg.h"
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
    perform_instance_allocation_for_value(DynamicNodeAttrs const &node,
                                          DynamicValueAttrs const &value,
                                          RealmContext &ctx) {
  ASSERT(value.accessor == std::nullopt);
  ASSERT(value.instance == std::nullopt);

  TensorShape shape = get_piece_shape(value.parallel_tensor_shape.value());

  MachineSpaceCoordinate device_coord = assert_unwrap(node.device_coord);
  Realm::Processor proc = ctx.map_device_coord_to_processor(device_coord);
  Realm::Memory memory = ctx.get_nearest_memory(proc);
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

  bidict<DynamicValueAttrs, DynamicValueAttrs> unallocated_to_allocated;
  auto allocate = [&](DynamicNodeAttrs const &n, DynamicValueAttrs const &v) {
    if (contains_key(preallocated, v)) {
      // FIXME: Attach external instance to existing allocation and use that
      NOT_IMPLEMENTED();
    } else {
      if (contains_key(unallocated_to_allocated, v)) {
        return unallocated_to_allocated.at_l(v);
      } else {
        DynamicValueAttrs v2 = perform_instance_allocation_for_value(n, v, ctx);
        uallocated_to_allocated.equate(v, v2);
      }
    }
  };

  DynamicOpenDataflowGraph result = transform_dynamic_invocation_set(
      g, [&](DynamicNodeInvocation const &i) -> DynamicNodeInvocation {
        return DynamicNodeInvocation{
            /*inputs=*/map_values(
                i.inputs,
                [&](DynamicValueAttrs const &v) -> DynamicValueAttrs {
                  return allocate(i.node_attrs, v);
                }),
            /*node_attrs=*/i.node_attrs,
            /*outputs=*/
            map_values(i.outputs,
                       [&](DynamicValueAttrs const &v) -> DynamicValueAttrs {
                         return allocate(i.node_attrs, v);
                       }),
        };
      });

  ASSERT(all_instances_are_allocated(result));

  return std::pair{result, ctx.get_outstanding_events()};
}

} // namespace FlexFlow
