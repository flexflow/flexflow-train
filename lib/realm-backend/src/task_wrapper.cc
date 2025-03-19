#include "realm-backend/task_wrapper.h"
#include <optional>
#include <unordered_set>

namespace FlexFlow {

using namespace Realm;


std::unordered_set<std::pair<int, task_id_t>> registered_tasks;

void init_wrapper_task(const void *args, size_t arglen, const void *userdata,
                       size_t userlen, Processor p) {
  RealmTaskArgs<DeviceSpecificDeviceStates> const &task_args =
      *reinterpret_cast<const RealmTaskArgs<DeviceSpecificDeviceStates> *>(args);
  auto fn =
      task_args.impl_function.get<InitOpTaskImplFunction>().function_ptr;
  DeviceSpecificDeviceStates result = fn(task_args.accessor);
  task_args.promise.set_value(result);
}

void fwdbwd_wrapper_task(const void *args, size_t arglen, const void *userdata,
                         size_t userlen, Processor p) {
  RealmTaskArgs<float> const &task_args =
      *reinterpret_cast<const RealmTaskArgs<float> *>(args);
  auto fn =
      task_args.impl_function.get<FwdBwdOpTaskImplFunction>().function_ptr;
  std::optional<float> result = fn(task_args.accessor);
  task_args.promise.set_value(result.has_value() ? result.value() : 0.0f);
}

void generic_wrapper_task(const void *args, size_t arglen, const void *userdata,
                          size_t userlen, Processor p) {
  RealmTaskArgs<void> const &task_args =
      *reinterpret_cast<const RealmTaskArgs<void> *>(args);
  auto fn =
      task_args.impl_function.get<GenericTaskImplFunction>().function_ptr;
  fn(task_args.accessor);
}

void register_wrapper_tasks_init(Processor p, task_id_t task_id) {
  Processor::register_task_by_kind(
      p.kind(), false /*!global*/, get_realm_task_id(task_id),
      CodeDescriptor(init_wrapper_task), ProfilingRequestSet())
      .external_wait();
}

void register_wrapper_tasks_fwdbwd(Realm::Processor p, task_id_t task_id) {
  Processor::register_task_by_kind(
      p.kind(), false /*!global*/, get_realm_task_id(task_id),
      CodeDescriptor(fwdbwd_wrapper_task), ProfilingRequestSet())
      .external_wait();
}

void register_wrapper_tasks_generic(Realm::Processor p, task_id_t task_id) {
  Processor::register_task_by_kind(
      p.kind(), false /*!global*/, get_realm_task_id(task_id),
      CodeDescriptor(generic_wrapper_task), ProfilingRequestSet())
      .external_wait();
}

void register_wrapper_tasks(int p_id, Processor p, task_id_t task_id,
                            TaskSignatureAndImpl task_sig_impl) {
  std::pair<int, task_id_t> key = {p_id, task_id};
  if (registered_tasks.find(key) != registered_tasks.end()) {
    return;
  }
  registered_tasks.insert(key);
  switch (task_sig_impl.task_signature.type) {
  case OpTaskType::INIT:
    register_wrapper_tasks_init(p, task_id);
    break;
  case OpTaskType::FWD:
  case OpTaskType::BWD:
    register_wrapper_tasks_fwdbwd(p, task_id);
    break;
  default:
    register_wrapper_tasks_generic(p, task_id);
    break;
  }
}

} // namespace FlexFlow