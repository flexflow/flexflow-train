#ifndef _FLEXFLOW_REALM_BACKEND_REALM_ARGS_BACKING_H
#define _FLEXFLOW_REALM_BACKEND_REALM_ARGS_BACKING_H

#include "pcg/computation_graph.h"
#include "pcg/layer_guid_t.dtg.h"
#include "realm-backend/realm_task_argument_accessor.h"
#include "realm-backend/task_result.h"
#include "task-spec/op_task_invocation.h"
#include "task-spec/per_device_op_state.h"
#include "task-spec/runtime_arg_config.h"
#include "task-spec/task_invocation.dtg.h"

namespace FlexFlow {

struct RealmArgsBacking {
  RealmArgsBacking(RuntimeArgConfig const &,
     std::unordered_map<layer_guid_t, DeviceSpecificDeviceStates> const &);

public:
  // arguments
  RuntimeArgConfig runtime_arg_config;
  std::unordered_map<layer_guid_t, DeviceSpecificDeviceStates>
      per_device_op_states;
};

RealmArgsBacking
make_args_backing_with_empty_device_states(RuntimeArgConfig const &);

std::optional<DeviceSpecificDeviceStates>
get_per_device_op_state_if_exists(RealmArgsBacking const &,
                                  layer_guid_t const &);

ArgSlotsBacking construct_arg_slots_backing(TaskBinding const &,
                                            RuntimeArgConfig const &);

} // namespace FlexFlow

#endif
