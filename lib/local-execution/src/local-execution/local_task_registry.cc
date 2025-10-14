#include "local-execution/local_task_registry.h"
#include "local-execution/operator_task_set.h"
#include "local-execution/registered_task.h"
#include "pcg/computation_graph.h"
#include "task-spec/task_signature_impl.h"
#include "utils/containers/contains_key.h"
#include "utils/containers/filtrans.h"
#include "utils/containers/flatmap.h"
#include "utils/containers/generate_map.h"
#include "utils/containers/map_values.h"
#include "utils/containers/try_at.h"
#include "utils/containers/values.h"

namespace FlexFlow {

LocalTaskRegistry construct_local_task_registry_for_layers(
    std::unordered_map<layer_guid_t, LayerAttrs> const &layer_attrs_mapping) {

  std::unordered_map<layer_guid_t, OperatorTaskSet> task_sets =
      map_values(layer_attrs_mapping, [](LayerAttrs const &layer_attrs) {
        return get_task_set_for_operator(layer_attrs.op_attrs);
      });

  std::unordered_set<registered_task_t> all_tasks =
      flatmap(unordered_set_of(values(task_sets)), get_all_tasks_in_task_set);

  std::unordered_set<task_id_t> all_real_tasks =
      filtrans(all_tasks, [](registered_task_t const &t) {
        return t.try_require_real_task();
      });

  std::unordered_map<task_id_t, TaskSignatureAndImpl> task_mapping =
      generate_map(all_real_tasks, get_task_signature_and_impl_for_task_id);

  return LocalTaskRegistry{
      /*task_sets=*/task_sets,
      /*task_mapping=*/task_mapping,
  };
}

std::optional<DeviceSpecificPerDeviceOpState> call_init_task_impl(
  LocalTaskRegistry const &local_task_registry,
  registered_task_t registered_task,
  TaskArgumentAccessor const &arg_accessor) {

  if (registered_task.is_noop_task()) {
    return std::nullopt;
  }

  task_id_t task_id = registered_task.require_real_task();

  TaskSignatureAndImpl task_sig_impl =
      local_task_registry.task_mapping.at(task_id);

  auto fn =
      task_sig_impl.impl_function.get<InitOpTaskImplFunction>().function_ptr;

  std::optional<DeviceSpecificPerDeviceOpState> device_state = fn(arg_accessor);

  return device_state;
}

std::optional<milliseconds_t>
    call_fwb_task_impl(LocalTaskRegistry const &task_registry,
                   task_id_t const &task_id,
                   TaskArgumentAccessor const &acc) {
  TaskSignatureAndImpl task_sig_impl = task_registry.task_mapping.at(task_id);
  auto fn =
      task_sig_impl.impl_function.get<FwdBwdOpTaskImplFunction>().function_ptr;

  return fn(acc);
}

void call_generic_task_impl(LocalTaskRegistry const &task_registry,
                            task_id_t const &task_id,
                            TaskArgumentAccessor const &acc) {
  TaskSignatureAndImpl task_sig_impl = task_registry.task_mapping.at(task_id);
  auto fn =
      task_sig_impl.impl_function.get<FwdBwdOpTaskImplFunction>().function_ptr;

  fn(acc);
}

} // namespace FlexFlow
