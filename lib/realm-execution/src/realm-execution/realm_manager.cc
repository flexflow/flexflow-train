#include "realm-execution/realm_manager.h"
#include "utils/exception.h"

namespace FlexFlow {

RealmManager::RealmManager(int *argc, char ***argv) {
  bool ok = this->runtime.init(argc, argv);
  ASSERT(ok);
}

void RealmManager::shutdown() {
  this->runtime.shutdown(this->last_event);
}

int RealmManager::wait_for_shutdown() {
  return this->runtime.wait_for_shutdown();
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

} // namespace FlexFlow
