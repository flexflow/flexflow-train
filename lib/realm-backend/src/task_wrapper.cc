#include "realm-backend/task_wrapper.h"

namespace FlexFlow {

using namespace Realm;

void init_wrapper_task(const void *args, size_t arglen, const void *userdata,
                       size_t userlen, Processor p) {
  RealmTaskArgs const &task_args =
      *reinterpret_cast<const RealmTaskArgs *>(args);
  auto fn =
      RealmTaskArgs.impl_function.get<InitOpTaskImplFunction>().function_ptr;
  *reinterpret_cast<DeviceSpecificDeviceStates *>(RealmTaskArgs.result) =
      fn(RealmTaskArgs.acc);
}

void fwdbwd_wrapper_task(const void *args, size_t arglen, const void *userdata,
                         size_t userlen, Processor p) {
  RealmTaskArgs const &task_args =
      *reinterpret_cast<const RealmTaskArgs *>(args);
  auto fn =
      RealmTaskArgs.impl_function.get<FwdBwdOpTaskImplFunction>().function_ptr;
  *reinterpret_cast<std::optional<float> *>(RealmTaskArgs.result) =
      fn(RealmTaskArgs.acc);
}

void generic_wrapper_task(const void *args, size_t arglen, const void *userdata,
                          size_t userlen, Processor p) {
  RealmTaskArgs const &task_args =
      *reinterpret_cast<const RealmTaskArgs *>(args);
  auto fn =
      RealmTaskArgs.impl_function.get<GenericTaskImplFunction>().function_ptr;
  fn(RealmTaskArgs.acc);
}

void register_wrapper_tasks_init(Processor p, task_id_t task_id) {
  Processor::register_task_by_kind(
      p.kind(), false /*!global*/, static_cast<Processor::TaskFuncID>(task_id),
      CodeDescriptor(init_wrapper_task), ProfilingRequestSet())
      .external_wait();
}

void register_wrapper_tasks_fwdbwd(Realm::Processor p, task_id_t task_id) {
  Processor::register_task_by_kind(
      p.kind(), false /*!global*/, static_cast<Processor::TaskFuncID>(task_id),
      CodeDescriptor(fwdbwd_wrapper_task), ProfilingRequestSet())
      .external_wait();
}

void register_wrapper_tasks_generic(Realm::Processor p, task_id_t task_id) {
  Processor::register_task_by_kind(
      p.kind(), false /*!global*/, static_cast<Processor::TaskFuncID>(task_id),
      CodeDescriptor(generic_wrapper_task), ProfilingRequestSet())
      .external_wait();
}

void register_wrapper_tasks(Processor p, task_id_t task_id,
                            TaskSignatureAndImpl task_sig_impl) {
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