#ifndef _FLEXFLOW_LIB_REALM_EXECUTION_INCLUDE_REALM_EXECUTION_REALM_CONTEXT_H
#define _FLEXFLOW_LIB_REALM_EXECUTION_INCLUDE_REALM_EXECUTION_REALM_CONTEXT_H

#include "kernels/allocation.h"
#include "kernels/device_handle_t.dtg.h"
#include "pcg/device_id_t.dtg.h"
#include "task-spec/realm/realm.h"

namespace FlexFlow {

struct RealmContext {
public:
  RealmContext();
  virtual ~RealmContext();

  RealmContext(RealmContext const &) = delete;
  RealmContext(RealmContext &&) = delete;

  // Current device context
  Allocator &get_current_device_allocator() const;
  device_handle_t const &get_current_device_handle() const;
  device_id_t const &get_current_device_idx() const;

protected:
  [[nodiscard]] Realm::Event merge_outstanding_events();

protected:
  Realm::Runtime runtime;
  std::vector<Realm::Event> outstanding_events;
};

} // namespace FlexFlow

#endif
