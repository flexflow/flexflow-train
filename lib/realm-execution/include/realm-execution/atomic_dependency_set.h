#ifndef _FLEXFLOW_LIB_REALM_EXECUTION_INCLUDE_REALM_EXECUTION_ATOMIC_DEPENDENCY_SET_H
#define _FLEXFLOW_LIB_REALM_EXECUTION_INCLUDE_REALM_EXECUTION_ATOMIC_DEPENDENCY_SET_H

#include "realm-execution/realm.h"
#include <vector>

namespace FlexFlow {

struct AtomicDependencySet {
public:
  AtomicDependencySet() = delete;
  explicit AtomicDependencySet(Realm::Event precondition);

  void add_writer(Realm::Event writer);
  void add_reader(Realm::Event reader);

  Realm::Event get_current_outstanding_events() const;

private:
  Realm::Event writer;
  std::vector<Realm::Event> readers;
};

} // namespace FlexFlow

#endif
