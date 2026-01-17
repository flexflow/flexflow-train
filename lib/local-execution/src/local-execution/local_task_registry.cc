#include "local-execution/local_task_registry.h"
#include "pcg/computation_graph.h"
#include "task-spec/task_impl_function.dtg.h"
#include "utils/containers/contains_key.h"
#include "utils/containers/filtrans.h"
#include "utils/containers/flatmap.h"
#include "utils/containers/generate_map.h"
#include "utils/containers/map_values.h"
#include "utils/containers/try_at.h"
#include "utils/containers/values.h"
#include "utils/exception.h"
#include "utils/optional.h"
#include <optional>

namespace FlexFlow {

std::optional<TaskImplFunction>
    get_init_task_impl_for_op_attrs(ComputationGraphOpAttrs const &op_attrs) {
  NOT_IMPLEMENTED();
}
TaskImplFunction
    get_fwd_task_impl_for_op_attrs(ComputationGraphOpAttrs const &op_attrs) {
  NOT_IMPLEMENTED();
}
TaskImplFunction
    get_bwd_task_impl_for_op_attrs(ComputationGraphOpAttrs const &op_attrs) {
  NOT_IMPLEMENTED();
}

std::optional<DeviceSpecificPerDeviceOpState>
    call_init_task_impl(ComputationGraphOpAttrs const &op_attrs,
                        TaskArgumentAccessor const &arg_accessor) {
  std::optional<TaskImplFunction> task_impl_fn =
      get_init_task_impl_for_op_attrs(op_attrs);
  if (!task_impl_fn) {
    return std::nullopt;
  }

  auto fn =
      assert_unwrap(task_impl_fn).get<InitOpTaskImplFunction>().function_ptr;

  std::optional<DeviceSpecificPerDeviceOpState> device_state = fn(arg_accessor);

  return device_state;
}

std::optional<milliseconds_t>
    call_fwb_task_impl(ComputationGraphOpAttrs const &op_attrs,
                       TaskArgumentAccessor const &acc) {

  TaskImplFunction task_impl_fn = get_fwd_task_impl_for_op_attrs(op_attrs);
  auto fn = task_impl_fn.get<FwdBwdOpTaskImplFunction>().function_ptr;

  return fn(acc);
}

void call_generic_task_impl(ComputationGraphOpAttrs const &op_attrs,
                            TaskArgumentAccessor const &acc) {

  NOT_IMPLEMENTED();
}

} // namespace FlexFlow
