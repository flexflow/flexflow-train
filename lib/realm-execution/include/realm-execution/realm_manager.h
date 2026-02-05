#ifndef _FLEXFLOW_LIB_REALM_EXECUTION_INCLUDE_REALM_EXECUTION_REALM_MANAGER_H
#define _FLEXFLOW_LIB_REALM_EXECUTION_INCLUDE_REALM_EXECUTION_REALM_MANAGER_H

#include "realm.h"

namespace FlexFlow {

struct RealmManager {
public:
  RealmManager(int *argc, char ***argv);

  RealmManager() = delete;
  RealmManager(RealmManager const &) = delete;
  RealmManager(RealmManager &&) = delete;

  Realm::Runtime get_runtime();
  void shutdown();
  int wait_for_shutdown();

private:
  Realm::Runtime runtime;
  Realm::Event last_event = Realm::Event::NO_EVENT;
};

} // namespace FlexFlow

#endif
