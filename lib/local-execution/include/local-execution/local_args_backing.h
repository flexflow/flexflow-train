#ifndef _FLEXFLOW_LOCAL_EXECUTION_LOCAL_ARGS_BACKING_H
#define _FLEXFLOW_LOCAL_EXECUTION_LOCAL_ARGS_BACKING_H

#include "local-execution/local_task_argument_accessor.h"
#include "pcg/computation_graph.h"
#include "pcg/layer_guid_t.dtg.h"
#include "task-spec/op_task_invocation.h"
#include "task-spec/per_device_op_state.h"
#include "task-spec/runtime_arg_config.h"
#include "task-spec/task_invocation.dtg.h"

namespace FlexFlow {

struct LocalArgsBacking {
  LocalArgsBacking(RuntimeArgConfig const &);

public:
  // arguments
  std::unordered_map<layer_guid_t, DeviceSpecificDeviceStates>
      per_device_op_states;
  RuntimeArgConfig runtime_arg_config;
};

void add_per_device_op_state(LocalArgsBacking &,
                             layer_guid_t const &,
                             DeviceSpecificDeviceStates const &);

std::optional<DeviceSpecificDeviceStates>
    get_per_device_op_state_if_exists(LocalArgsBacking const &,
                                      layer_guid_t const &);

ArgSlotsBacking construct_arg_slots_backing(TaskBinding const &,
                                            RuntimeArgConfig const &);

} // namespace FlexFlow

#endif
