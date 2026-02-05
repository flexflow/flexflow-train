#ifndef _FLEXFLOW_LIB_REALM_EXECUTION_INCLUDE_REALM_EXECUTION_REALM_MANAGER_H
#define _FLEXFLOW_LIB_REALM_EXECUTION_INCLUDE_REALM_EXECUTION_REALM_MANAGER_H

#include "kernels/allocation.h"
#include "kernels/device_handle_t.dtg.h"
#include "pcg/device_id_t.dtg.h"
#include "realm.h"

namespace FlexFlow {

struct RealmManager {
public:
  RealmManager(int *argc, char ***argv);

  RealmManager() = delete;
  RealmManager(RealmManager const &) = delete;
  RealmManager(RealmManager &&) = delete;

  void shutdown();
  int wait_for_shutdown();

  Allocator &get_current_device_allocator() const;

  device_handle_t const &get_current_device_handle() const;
  device_id_t const &get_current_device_idx() const;

private:
  Realm::Runtime runtime;
  Realm::Event last_event = Realm::Event::NO_EVENT;
};

} // namespace FlexFlow

#endif
