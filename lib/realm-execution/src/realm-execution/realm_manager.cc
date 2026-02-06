#include "realm-execution/realm_manager.h"
#include "realm-execution/realm_task_id_t.h"
#include "realm-execution/task_id_t.dtg.h"
#include "utils/exception.h"

namespace FlexFlow {

RealmManager::RealmManager(int *argc, char ***argv) {
  bool ok = this->runtime.init(argc, argv);
  ASSERT(ok);
}

RealmManager::~RealmManager() {
  Realm::Event outstanding = this->merge_outstanding_events();
  this->runtime.shutdown(outstanding);
  this->runtime.wait_for_shutdown();
}

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

Realm::Event
    RealmManager::start_controller(std::function<void(RealmContext &)> thunk) {
  Realm::Processor::TaskFuncID CONTROLLER_TASK_ID =
      get_realm_task_id_for_task_id(task_id_t::CONTROLLER_TASK_ID);
  Realm::Event task_ready = Realm::Processor::register_task_by_kind(
      Realm::Processor::LOC_PROC,
      /*global=*/false,
      CONTROLLER_TASK_ID,
      Realm::CodeDescriptor(controller_task_wrapper),
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

} // namespace FlexFlow
