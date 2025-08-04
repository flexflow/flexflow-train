#include "realm-backend/task_wrapper.h"
#include <optional>
#include <unordered_set>

namespace FlexFlow {

using namespace Realm;


std::unordered_set<std::pair<int, task_id_t>> registered_tasks;

void init_wrapper_task(const void *args, size_t arglen, const void *userdata,
                       size_t userlen, Processor p) {
  assert(arglen == sizeof(uintptr_t));
  uintptr_t task_arg_ptr = *reinterpret_cast<const uintptr_t *>(args);
  RealmTaskArgs<DeviceSpecificDeviceStates> *task_args =
      reinterpret_cast<RealmTaskArgs<DeviceSpecificDeviceStates> *>(task_arg_ptr);
  auto fn =
      task_args->impl_function.get<InitOpTaskImplFunction>().function_ptr;
  DeviceSpecificDeviceStates result = fn(task_args->accessor);
  task_args->promise.set_value(result);
  delete task_args;
}

void fwdbwd_wrapper_task(const void *args, size_t arglen, const void *userdata,
                         size_t userlen, Processor p) {
  assert(arglen == sizeof(uintptr_t));
  uintptr_t task_arg_ptr = *reinterpret_cast<const uintptr_t *>(args);
  RealmTaskArgs<std::optional<milliseconds_t>> *task_args =
      reinterpret_cast<RealmTaskArgs<std::optional<milliseconds_t>> *>(task_arg_ptr);
  auto fn =
      task_args->impl_function.get<FwdBwdOpTaskImplFunction>().function_ptr;
  std::optional<milliseconds_t> result = transform(
      fn(task_args->accessor), [](float running_time) { return milliseconds_t{running_time}; });
  task_args->promise.set_value(std::move(result));
  delete task_args;
}

void generic_wrapper_task(const void *args, size_t arglen, const void *userdata,
                          size_t userlen, Processor p) {
  assert(arglen == sizeof(uintptr_t));
  uintptr_t task_arg_ptr = *reinterpret_cast<const uintptr_t *>(args);
  RealmTaskArgs<void> *task_args =
      reinterpret_cast<RealmTaskArgs<void> *>(task_arg_ptr);
  auto fn =
      task_args->impl_function.get<GenericTaskImplFunction>().function_ptr;
  fn(task_args->accessor);
  delete task_args;
}

void register_wrapper_tasks_init(int p_id, Processor p, task_id_t task_id) {
  std::pair<int, task_id_t> key = {p_id, task_id};
  if (registered_tasks.find(key) != registered_tasks.end()) {
    return;
  }
  registered_tasks.insert(key);
  Processor::register_task_by_kind(
      p.kind(), false /*!global*/, get_realm_task_id(task_id),
      CodeDescriptor(init_wrapper_task), ProfilingRequestSet())
      .external_wait();
}

void register_wrapper_tasks_fwdbwd(int p_id, Realm::Processor p, task_id_t task_id) {
  std::pair<int, task_id_t> key = {p_id, task_id};
  if (registered_tasks.find(key) != registered_tasks.end()) {
    return;
  }
  registered_tasks.insert(key);
  Processor::register_task_by_kind(
      p.kind(), false /*!global*/, get_realm_task_id(task_id),
      CodeDescriptor(fwdbwd_wrapper_task), ProfilingRequestSet())
      .external_wait();
}

void register_wrapper_tasks_generic(int p_id, Realm::Processor p, task_id_t task_id) {
  std::pair<int, task_id_t> key = {p_id, task_id};
  if (registered_tasks.find(key) != registered_tasks.end()) {
    return;
  }
  registered_tasks.insert(key);
  Processor::register_task_by_kind(
      p.kind(), false /*!global*/, get_realm_task_id(task_id),
      CodeDescriptor(generic_wrapper_task), ProfilingRequestSet())
      .external_wait();
}

void register_wrapper_tasks(int p_id, Processor p, task_id_t task_id,
                            TaskSignatureAndImpl task_sig_impl) {
  switch (task_sig_impl.task_signature.type) {
  case OpTaskType::INIT:
    register_wrapper_tasks_init(p_id, p, task_id);
    break;
  case OpTaskType::FWD:
  case OpTaskType::BWD:
    register_wrapper_tasks_fwdbwd(p_id, p, task_id);
    break;
  default:
    register_wrapper_tasks_generic(p_id, p, task_id);
    break;
  }
}

} // namespace FlexFlow