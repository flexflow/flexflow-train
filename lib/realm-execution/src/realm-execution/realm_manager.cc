#include "realm-execution/realm_manager.h"
#include "realm-execution/realm_context.h"
#include "realm-execution/tasks/impl/controller_task.h"
#include "realm-execution/tasks/realm_task_registry.h"

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

  return collective_spawn_controller_task(*this, target_proc, thunk);
}

} // namespace FlexFlow
