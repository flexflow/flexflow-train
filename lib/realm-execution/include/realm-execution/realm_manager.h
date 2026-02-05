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
  ~RealmManager();

  RealmManager() = delete;
  RealmManager(RealmManager const &) = delete;
  RealmManager(RealmManager &&) = delete;

  Realm::Event start_controller(void (*thunk)(RealmManager &));

  // Current device context
  Allocator &get_current_device_allocator() const;
  device_handle_t const &get_current_device_handle() const;
  device_id_t const &get_current_device_idx() const;

private:
  RealmManager(void const *, size_t, void const *, size_t, Realm::Processor);

  [[nodiscard]] Realm::Event merge_outstanding_events();

  static void controller_task_wrapper(
      void const *, size_t, void const *, size_t, Realm::Processor);

private:
  Realm::Runtime runtime;
  std::vector<Realm::Event> outstanding_events;
  bool is_root_runtime;
};

} // namespace FlexFlow

#endif
