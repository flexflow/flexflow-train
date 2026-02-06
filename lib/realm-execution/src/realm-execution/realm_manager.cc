#include "realm-execution/realm_manager.h"
#include "realm-execution/realm_task_id_t.h"
#include "realm-execution/realm_task_registry.h"
#include "realm-execution/task_id_t.dtg.h"
#include "utils/exception.h"

namespace FlexFlow {

static void controller_task_wrapper(void const *args,
                                    size_t arglen,
                                    void const *userdata,
                                    size_t userlen,
                                    Realm::Processor proc) {
  ASSERT(arglen == sizeof(std::function<void(RealmContext &)>));
  std::function<void(RealmContext &)> thunk =
      *reinterpret_cast<std::function<void(RealmContext &)> const *>(args);

  RealmContext ctx;
  thunk(ctx);
}

RealmManager::RealmManager(int *argc, char ***argv) {
  bool ok = this->runtime.init(argc, argv);
  ASSERT(ok);

  // Register all tasks at initialization time so we don't need to later
  register_all_tasks().wait();
  register_task(Realm::Processor::LOC_PROC,
                task_id_t::CONTROLLER_TASK_ID,
                controller_task_wrapper)
      .wait();
}

RealmManager::~RealmManager() {
  Realm::Event outstanding = this->merge_outstanding_events();
  this->runtime.shutdown(outstanding);
  this->runtime.wait_for_shutdown();
}

Realm::Event
    RealmManager::start_controller(std::function<void(RealmContext &)> thunk) {
  Realm::Processor target_proc =
      Realm::Machine::ProcessorQuery(Realm::Machine::get_machine())
          .only_kind(Realm::Processor::LOC_PROC)
          .first();

  Realm::Event task_complete = this->runtime.collective_spawn(
      target_proc,
      get_realm_task_id_for_task_id(task_id_t::CONTROLLER_TASK_ID),
      &thunk,
      sizeof(thunk));
  this->outstanding_events.push_back(task_complete);
  return task_complete;
}

} // namespace FlexFlow
