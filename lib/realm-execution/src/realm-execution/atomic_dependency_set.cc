#include "realm-execution/atomic_dependency_set.h"

namespace FlexFlow {

AtomicDependencySet::AtomicDependencySet(Realm::Event precondition)
    : writer(precondition) {}

void AtomicDependencySet::add_writer(Realm::Event writer) {
  this->writer = Realm::Event::merge_events(
      writer, this->get_current_outstanding_events());
  this->readers.clear();
}

void AtomicDependencySet::add_reader(Realm::Event reader) {
  this->readers.push_back(reader);
}

Realm::Event AtomicDependencySet::get_current_outstanding_events() const {
  Realm::Event readers = Realm::Event::merge_events(this->readers);
  return Realm::Event::merge_events(writer, readers);
}

} // namespace FlexFlow
