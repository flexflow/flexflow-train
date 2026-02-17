#include "realm-execution/distributed_device_state_initialization.h"
#include "local-execution/device_state_initialization.h"
#include "realm-execution/tasks/impl/device_state_init_task.h"
#include "realm-execution/tensor_instance_backing.dtg.h"
#include "realm-execution/tensor_instance_backing.h"
#include "task-spec/dynamic_graph/dynamic_node_invocation.dtg.h"
#include "task-spec/dynamic_graph/dynamic_open_dataflow_graph.h"
#include "task-spec/dynamic_graph/dynamic_value_attrs.dtg.h"
#include "utils/containers/map_values.h"
#include "utils/optional.h"
#include <optional>
#include <unordered_map>

namespace FlexFlow {

DynamicOpenDataflowGraph perform_distributed_device_state_initialization(
    RealmContext &ctx,
    DynamicOpenDataflowGraph const &dg,
    TensorInstanceBacking const &tensor_instance_backing,
    ProfilingSettings const &profiling_settings,
    DistributedDeviceHandle const &device_handle,
    FFIterationConfig const &iteration_config,
    OptimizerAttrs const &optimizer_attrs,
    Realm::Event precondition) {

  // Initialize all operators and save the per-device op state
  ASSERT(no_nodes_are_initialized(dg));

  std::unordered_map<DynamicNodeInvocation, DeviceSpecificPerDeviceOpState *>
      result_map;
  for (DynamicNodeInvocation const &invocation : dg.invocations) {
    Realm::Processor target_proc = ctx.map_device_coord_to_processor(
        assert_unwrap(invocation.node_attrs.device_coord));

    TensorInstanceBacking tensor_backing =
        subset_tensor_instance_backing_for_invocation(tensor_instance_backing,
                                                      invocation);

    // FIXME: in the absense of a real serializer we're just tossing around raw
    // bytes, which means we need to bypass the constructor for this type (yes,
    // ugh)
    DeviceSpecificPerDeviceOpState *output =
        static_cast<DeviceSpecificPerDeviceOpState *>(
            malloc(sizeof(DeviceSpecificPerDeviceOpState)));
    std::optional<Realm::Event> result =
        spawn_device_state_init_task(ctx,
                                     target_proc,
                                     invocation,
                                     tensor_backing,
                                     profiling_settings,
                                     device_handle.at(target_proc),
                                     iteration_config,
                                     optimizer_attrs,
                                     output,
                                     precondition);
    if (result) {
      result_map[invocation] = output;
    } else {
      free(output);
    }
  }

  ctx.get_outstanding_events().wait();

  DynamicOpenDataflowGraph result = transform_dynamic_invocation_set(
      dg, [&](DynamicNodeInvocation const &invocation) {
        DynamicNodeInvocation result = invocation;
        auto device_state = result_map.find(invocation);
        if (device_state != result_map.end()) {
          result.node_attrs.per_device_op_state = *device_state->second;
        }
        return result;
      });

  for (auto &[invocation, output] : result_map) {
    free(output);
  }

  return result;
}

} // namespace FlexFlow
