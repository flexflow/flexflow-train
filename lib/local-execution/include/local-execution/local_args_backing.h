#ifndef _FLEXFLOW_LOCAL_EXECUTION_LOCAL_ARGS_BACKING_H
#define _FLEXFLOW_LOCAL_EXECUTION_LOCAL_ARGS_BACKING_H

#include "local-execution/local_task_argument_accessor.h"
#include "local-execution/op_task_invocation.h"
#include "local-execution/per_device_op_state.h"
#include "local-execution/runtime_arg_config.h"
#include "local-execution/task_invocation.dtg.h"
#include "pcg/computation_graph.h"
#include "pcg/layer_guid_t.dtg.h"

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
