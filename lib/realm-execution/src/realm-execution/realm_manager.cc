#include "realm-execution/realm_manager.h"
#include "realm-execution/realm_context.h"
#include "realm-execution/tasks/realm_task_id_t.h"
#include "realm-execution/tasks/realm_task_registry.h"
#include "realm-execution/tasks/realm_tasks.h"
#include "realm-execution/tasks/task_id_t.dtg.h"
#include "utils/exception.h"

namespace FlexFlow {

RealmManager::RealmManager(int *argc, char ***argv)
    : RealmContext(Realm::Processor::NO_PROC) {
  bool ok = this->runtime.init(argc, argv);
  ASSERT(ok);

  // Register all tasks at initialization time so we don't need to later
  register_all_tasks().wait();
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

  ControllerTaskArgs task_args;
  task_args.thunk = thunk;
  Realm::Event task_complete = this->runtime.collective_spawn(
      target_proc,
      get_realm_task_id_for_task_id(task_id_t::CONTROLLER_TASK_ID),
      &task_args,
      sizeof(task_args));
  this->outstanding_events.push_back(task_complete);
  return task_complete;
}

} // namespace FlexFlow
