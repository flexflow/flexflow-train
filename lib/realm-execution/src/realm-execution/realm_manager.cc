#include "realm-execution/realm_manager.h"
#include "utils/exception.h"

namespace FlexFlow {

RealmManager::RealmManager(int *argc, char ***argv) {
  bool ok = this->runtime.init(argc, argv);
  ASSERT(ok);
}

Realm::Runtime RealmManager::get_runtime() {
  return this->runtime;
}

void RealmManager::shutdown() {
  this->runtime.shutdown(this->last_event);
}

int RealmManager::wait_for_shutdown() {
  return this->runtime.wait_for_shutdown();
}
} // namespace FlexFlow
