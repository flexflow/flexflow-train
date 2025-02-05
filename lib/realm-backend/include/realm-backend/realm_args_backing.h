#ifndef _FLEXFLOW_REALM_BACKEND_REALM_ARGS_BACKING_H
#define _FLEXFLOW_REALM_BACKEND_REALM_ARGS_BACKING_H

#include "local-execution/op_task_invocation.h"
#include "local-execution/per_device_op_state.h"
#include "local-execution/runtime_arg_config.h"
#include "local-execution/task_invocation.dtg.h"
#include "pcg/computation_graph.h"
#include "pcg/layer_guid_t.dtg.h"
#include "realm-backend/realm_task_argument_accessor.h"
#include "realm-backend/task_result.h"

namespace FlexFlow {

struct RealmArgsBacking {
  RealmArgsBacking(RuntimeArgConfig const &);

public:
  void add_per_device_op_state(layer_guid_t const &,
                               Future<DeviceSpecificDeviceStates> &&);

  ArgSlotsBacking construct_arg_slots_backing(TaskBinding const &) const;

  ConcreteArgSpec lower_to_concrete_arg_spec(RuntimeArgRefSpec const &) const;
  ConcreteArgSpec lower_to_concrete_arg_spec(OpArgRefSpec const &,
                                             ComputationGraph const &,
                                             layer_guid_t const &) const;

public:
  // arguments
  std::unordered_map<layer_guid_t, DeviceSpecificDeviceStates>
      per_device_op_states;
  RuntimeArgConfig runtime_arg_config;
};

} // namespace FlexFlow

#endif
