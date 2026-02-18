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

PerDeviceOpStateBacking perform_distributed_device_state_initialization(
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

  std::unordered_map<DynamicNodeInvocation, DeviceSpecificPtr<PerDeviceOpState>>
      result;

  // Preallocate output before launching tasks
  for (DynamicNodeInvocation const &invocation : dg.invocations) {
    result.insert(std::pair{invocation,
                            DeviceSpecificPtr<PerDeviceOpState>{
                                ctx.get_current_device_idx(), std::nullopt}});
  }

  for (DynamicNodeInvocation const &invocation : dg.invocations) {
    Realm::Processor target_proc = ctx.map_device_coord_to_processor(
        assert_unwrap(invocation.node_attrs.device_coord));

    TensorInstanceBacking tensor_backing =
        subset_tensor_instance_backing_for_invocation(tensor_instance_backing,
                                                      invocation);

    spawn_device_state_init_task(ctx,
                                 target_proc,
                                 invocation,
                                 tensor_backing,
                                 profiling_settings,
                                 device_handle.at(target_proc),
                                 iteration_config,
                                 optimizer_attrs,
                                 &result.at(invocation),
                                 precondition);
  }

  ctx.get_outstanding_events().wait();

  return PerDeviceOpStateBacking{/*backing=*/result};
}

} // namespace FlexFlow
