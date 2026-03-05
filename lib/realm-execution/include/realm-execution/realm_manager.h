#ifndef _FLEXFLOW_LIB_REALM_EXECUTION_INCLUDE_REALM_EXECUTION_REALM_MANAGER_H
#define _FLEXFLOW_LIB_REALM_EXECUTION_INCLUDE_REALM_EXECUTION_REALM_MANAGER_H

#include "kernels/allocation.h"
#include "kernels/device_handle_t.dtg.h"
#include "pcg/device_id_t.dtg.h"
#include "realm-execution/realm.h"
#include "realm-execution/realm_context.h"

namespace FlexFlow {

/**
 * @brief Manages the initialization and shutdown of the realm runtime.
 * Provides the interface to launch the \ref term-controller that runs the rest of the computation
* (i.e., \ref start_controller).
 */
struct RealmManager : private RealmContext {
public:
  RealmManager(int *argc, char ***argv);
  virtual ~RealmManager();

  RealmManager() = delete;
  RealmManager(RealmManager const &) = delete;
  RealmManager(RealmManager &&) = delete;

  [[nodiscard]] Realm::Event
      start_controller(std::function<void(RealmContext &)>,
                       Realm::Event wait_on = Realm::Event::NO_EVENT);
};

} // namespace FlexFlow

#endif
