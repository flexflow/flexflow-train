#include "realm-execution/instance_allocation.h"
#include "local-execution/tensor_allocation.h"
#include "op-attrs/num_ptensor_shard_dims_t.dtg.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "op-attrs/relative_ff_dim_t.h"
#include "op-attrs/shard_parallel_dim.dtg.h"
#include "op-attrs/tensor_shape.dtg.h"
#include "realm-execution/realm_context.h"
#include "realm-execution/tensor_instance_backing.h"
#include "task-spec/dynamic_graph/dynamic_node_attrs.dtg.h"
#include "task-spec/dynamic_graph/dynamic_node_invocation.dtg.h"
#include "task-spec/dynamic_graph/dynamic_open_dataflow_graph.h"
#include "task-spec/dynamic_graph/dynamic_tensor_accessor.dtg.h"
#include "task-spec/dynamic_graph/dynamic_value_attrs.dtg.h"
#include "utils/bidict/generate_bidict.h"
#include "utils/containers/all_are_true.h"
#include "utils/containers/contains_key.h"
#include "utils/containers/make.h"
#include "utils/containers/map_values.h"
#include "utils/containers/unordered_set_of.h"
#include "utils/containers/values.h"
#include "utils/exception.h"
#include "utils/nonnegative_int/nonnegative_int.h"
#include "utils/optional.h"
#include "utils/overload.h"

namespace FlexFlow {
std::pair<Realm::RegionInstance, Realm::Event>
    perform_instance_allocation_for_value(
        MachineSpaceCoordinate const &device_coord,
        DynamicValueAttrs const &value,
        RealmContext &ctx) {
  ASSERT(value.accessor == std::nullopt);

  ParallelTensorShape const par_shape = value.parallel_tensor_shape.value();

  TensorShape shape = get_per_device_shape(par_shape);

  Realm::Processor proc = ctx.map_device_coord_to_processor(device_coord);
  Realm::Memory memory = ctx.get_nearest_memory(proc);

  int ndims = static_cast<int>(num_shard_dims(par_shape).value);
  std::vector<int> offsets(ndims, 0);

  if (value.shard_coord.has_value()) {
    ParallelTensorSpaceCoordinate const &coord = value.shard_coord.value();

    for (int i = 0; i < ndims; i++) {
      relative_ff_dim_t rel_dim{i};

      // skip if shard_components doesn't have this dim
      if (!coord.shard_components.idx_is_valid(rel_dim)) {
        continue;
      }

      ShardParallelDim shard_dim = par_shape.dims.shard_dims.at(rel_dim);

      // skip if not actually sharded
      if (shard_dim.degree == 1_p) {
        continue;
      }

      nonnegative_int piece_size =
          shard_dim.size.nonnegative_int_from_positive_int() /
          shard_dim.degree.nonnegative_int_from_positive_int();
      nonnegative_int shard_idx = coord.shard_components.at(rel_dim);
      offsets[i] = static_cast<int>(shard_idx * piece_size);
    }
  }

  bool has_offset =
      std::any_of(offsets.begin(), offsets.end(), [](int o) { return o != 0; });

  if (has_offset) {
    return ctx.create_instance_with_offset(
        memory, shape, offsets, Realm::ProfilingRequestSet());
  } else {
    return ctx.create_instance(memory, shape, Realm::ProfilingRequestSet());
  }
}
TensorInstanceBacking perform_instance_allocation(
    DynamicOpenDataflowGraph const &g,
    std::unordered_map<DynamicValueAttrs, DynamicTensorAccessor> const
        &preallocated,
    std::unordered_map<DynamicValueAttrs,
                       std::pair<Realm::RegionInstance, Realm::Event>> const
        &preallocated_instances,
    RealmContext &ctx) {

  ASSERT(no_tensors_are_allocated(g));
  ASSERT(tensors_are_ready_for_allocation(g));
  for (DynamicValueAttrs const &v : keys(preallocated)) {
    ASSERT(v.accessor == std::nullopt);
  }

  TensorInstanceBacking result = make_empty_tensor_instance_backing();
  auto allocate = [&](DynamicNodeAttrs const &n, DynamicValueAttrs const &v) {
    // check pre-created instances first
    if (contains_key(preallocated_instances, v)) {
      if (!contains_key(result.backing, v)) {
        result.backing.insert(std::make_pair(v, preallocated_instances.at(v)));
      }
      return result.backing.at(v);
    }

    // then check accessor-based preallocated
    if (contains_key(preallocated, v)) {
      if (!contains_key(result.backing, v)) {
        DynamicTensorAccessor const &accessor = preallocated.at(v);

        void *ptr = accessor.visit<void *>(overload{
            [](GenericTensorAccessorR const &a) {
              return const_cast<void *>(a.ptr);
            },
            [](GenericTensorAccessorW const &a) { return a.ptr; },
        });

        MachineSpaceCoordinate device_coord = assert_unwrap(n.device_coord);
        Realm::Processor proc = ctx.map_device_coord_to_processor(device_coord);
        Realm::Memory memory = ctx.get_nearest_memory(proc);

        ParallelTensorShape const &par_shape = v.parallel_tensor_shape.value();
        TensorShape shape = get_per_device_shape(par_shape);

        int ndims = static_cast<int>(num_shard_dims(par_shape).value);
        std::vector<int> offsets(ndims, 0);
        if (v.shard_coord.has_value()) {
          ParallelTensorSpaceCoordinate const &coord = v.shard_coord.value();
          for (int i = 0; i < ndims; i++) {
            relative_ff_dim_t rel_dim{i};
            if (!coord.shard_components.idx_is_valid(rel_dim)) {
              continue;
            }
            ShardParallelDim shard_dim = par_shape.dims.shard_dims.at(rel_dim);
            if (shard_dim.degree == 1_p) {
              continue;
            }
            nonnegative_int piece_size =
                shard_dim.size.nonnegative_int_from_positive_int() /
                shard_dim.degree.nonnegative_int_from_positive_int();
            nonnegative_int shard_idx = coord.shard_components.at(rel_dim);
            offsets[i] = static_cast<int>(shard_idx * piece_size);
          }
        }

        result.backing.insert(std::pair{
            v,
            ctx.create_external_instance(
                memory, shape, offsets, ptr, Realm::ProfilingRequestSet())});
      }
      return result.backing.at(v);
    } else {
      if (!contains_key(result.backing, v)) {
        MachineSpaceCoordinate device_coord = assert_unwrap(n.device_coord);
        result.backing.insert(std::pair{
            v, perform_instance_allocation_for_value(device_coord, v, ctx)});
      }
      return result.backing.at(v);
    }
  };

  for (DynamicNodeInvocation const &invocation : g.invocations) {
    for (DynamicValueAttrs const &input : values(invocation.inputs)) {
      allocate(invocation.node_attrs, input);
    }
    for (DynamicValueAttrs const &output : values(invocation.outputs)) {
      allocate(invocation.node_attrs, output);
    }
  }

  return result;
}

void destroy_instances(TensorInstanceBacking const &instances,
                       Realm::Event precondition) {
  for (auto const &[instance, ready] : values(instances.backing)) {
    instance.destroy(Realm::Event::merge_events(precondition, ready));
  }
}

} // namespace FlexFlow
