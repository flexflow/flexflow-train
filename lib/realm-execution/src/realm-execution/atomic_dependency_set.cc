#include "realm-execution/atomic_dependency_set.h"

namespace FlexFlow {

AtomicDependencySet::AtomicDependencySet(Realm::Event precondition)
    : writer(precondition) {}

void AtomicDependencySet::add_writer(Realm::Event writer) {
  this->writer =
      Realm::Event::merge_events(writer, this->get_dependency_for_writer());
  this->readers.clear();
}

void AtomicDependencySet::add_reader(Realm::Event reader) {
  this->readers.push_back(reader);
}

Realm::Event AtomicDependencySet::get_dependency_for_writer() const {
  Realm::Event readers = Realm::Event::merge_events(this->readers);
  return Realm::Event::merge_events(this->writer, readers);
}

Realm::Event AtomicDependencySet::get_dependency_for_reader() const {
  return this->writer;
}

} // namespace FlexFlow
