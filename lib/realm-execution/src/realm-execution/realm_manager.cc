#include "realm-execution/realm_manager.h"
#include "utils/exception.h"

namespace FlexFlow {

RealmManager::RealmManager(int *argc, char ***argv) : is_root_runtime(true) {
  bool ok = this->runtime.init(argc, argv);
  ASSERT(ok);
}

RealmManager::RealmManager(void const *args,
                           size_t arglen,
                           void const *userdata,
                           size_t userdatalen,
                           Realm::Processor proc)
    : runtime(Realm::Runtime::get_runtime()), is_root_runtime(false) {}

RealmManager::~RealmManager() {
  Realm::Event outstanding = this->merge_outstanding_events();
  if (is_root_runtime) {
    this->runtime.shutdown(outstanding);
    this->runtime.wait_for_shutdown();
  } else {
    outstanding.wait();
  }
}

Realm::Event RealmManager::start_controller(void (*thunk)(RealmManager &)) {
  constexpr int CONTROLLER_TASK_ID = Realm::Processor::TASK_ID_FIRST_AVAILABLE;
  Realm::Event task_ready = Realm::Processor::register_task_by_kind(
      Realm::Processor::LOC_PROC,
      /*global=*/false,
      CONTROLLER_TASK_ID,
      Realm::CodeDescriptor(RealmManager::controller_task_wrapper),
      Realm::ProfilingRequestSet(),
      &thunk,
      sizeof(thunk));

  Realm::Processor target_proc =
      Realm::Machine::ProcessorQuery(Realm::Machine::get_machine())
          .only_kind(Realm::Processor::LOC_PROC)
          .first();

  Realm::Event task_complete = this->runtime.collective_spawn(
      target_proc, CONTROLLER_TASK_ID, &thunk, sizeof(thunk), task_ready);
  this->outstanding_events.push_back(task_complete);
  return task_complete;
}

Allocator &RealmManager::get_current_device_allocator() const {
  NOT_IMPLEMENTED();
}

device_handle_t const &RealmManager::get_current_device_handle() const {
  NOT_IMPLEMENTED();
}
device_id_t const &RealmManager::get_current_device_idx() const {
  NOT_IMPLEMENTED();
}

Realm::Event RealmManager::merge_outstanding_events() {
  Realm::Event result = Realm::Event::merge_events(this->outstanding_events);
  this->outstanding_events.clear();
  return result;
}

void RealmManager::controller_task_wrapper(void const *args,
                                           size_t arglen,
                                           void const *userdata,
                                           size_t userlen,
                                           Realm::Processor proc) {
  assert(arglen == sizeof(void (*)(RealmManager &)));
  void (*thunk)(RealmManager &) =
      *reinterpret_cast<void (*const *)(RealmManager &)>(args);

  RealmManager manager(args, arglen, userdata, userlen, proc);
  thunk(manager);
}

} // namespace FlexFlow
