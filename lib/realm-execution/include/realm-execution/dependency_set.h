#ifndef _FLEXFLOW_LIB_REALM_EXECUTION_INCLUDE_REALM_EXECUTION_DEPENDENCY_SET_H
#define _FLEXFLOW_LIB_REALM_EXECUTION_INCLUDE_REALM_EXECUTION_DEPENDENCY_SET_H

#include "realm-execution/atomic_dependency_set.h"
#include "realm-execution/realm.h"
#include "task-spec/dynamic_graph/dynamic_value_attrs.dtg.h"
#include <unordered_map>

namespace FlexFlow {

struct DependencySet {
public:
  DependencySet() = delete;
  explicit DependencySet(Realm::Event precondition);

  void add_writer(DynamicValueAttrs const &value, Realm::Event writer);
  void add_reader(DynamicValueAttrs const &value, Realm::Event reader);

  Realm::Event get_dependency_for_writer(DynamicValueAttrs const &value) const;
  Realm::Event get_dependency_for_reader(DynamicValueAttrs const &value) const;

private:
  AtomicDependencySet &
      get_atomic_dependency_set(DynamicValueAttrs const &value);

private:
  Realm::Event precondition;
  std::unordered_map<DynamicValueAttrs, AtomicDependencySet>
      atomic_dependencies;
};

} // namespace FlexFlow

#endif
