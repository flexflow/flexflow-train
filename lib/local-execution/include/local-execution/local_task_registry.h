#ifndef _FLEXFLOW_LOCAL_EXECUTION_TASK_REGISTRY_H
#define _FLEXFLOW_LOCAL_EXECUTION_TASK_REGISTRY_H

#include "local-execution/local_task_registry.dtg.h"
#include "local-execution/registered_task_t.dtg.h"
#include "pcg/layer_attrs.dtg.h"
#include "task-spec/op_task_type.dtg.h"
#include "utils/units/milliseconds_t.h"
#include "task-spec/device_specific_per_device_op_state.dtg.h"

namespace FlexFlow {

LocalTaskRegistry construct_local_task_registry_for_layers(
    std::unordered_map<layer_guid_t, LayerAttrs> const &);

std::optional<registered_task_t> try_get_registered_task(
    LocalTaskRegistry const &, layer_guid_t const &, OpTaskType const &);

std::optional<DeviceSpecificPerDeviceOpState> call_init_task_impl(
  LocalTaskRegistry const &local_task_registry,
  registered_task_t task_id,
  TaskArgumentAccessor const &arg_accessor);

std::optional<milliseconds_t> call_fwb_task_impl(
  LocalTaskRegistry const &local_task_registry,
  task_id_t task_id,
  TaskArgumentAccessor const &arg_accessor);

} // namespace FlexFlow

#endif
