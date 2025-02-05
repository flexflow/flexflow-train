#include "realm-backend/task_result.h"

namespace FlexFlow {

/************ SharedState implementation ************/
template <typename T> SharedState<T>::SharedState(Realm::Memory mem) {
  Realm::Rect<1> bounds(Realm::Point<1>(0), Realm::Point<1>(0));
  this->inst = Realm::RegionInstance::NO_INST;
  Realm::RegionInstance::create_instance(
      this->inst, mem, bounds, {sizeof(T)}, /*SOA*/ 1,
      Realm::ProfilingRequestSet(), Realm::Event::NO_EVENT)
      .wait();
}

template <typename T> void SharedState<T>::set_event(Realm::Event e) {
  this->event = e;
}

template <typename T> void SharedState<T>::set_value(T &&value) {
  Realm::GenericAccessor<T, 1> acc(this->inst, 0);
  acc[Realm::Point<1>(0)] = std::move(value);
}

template <typename T> void SharedState<T>::wait() { this->event.wait(); }

template <typename T> T SharedState<T>::get_value() {
  wait();
  Realm::GenericAccessor<T, 1> acc(this->inst, 0);
  return acc[Realm::Point<1>(0)];
}

void SharedState<void>::set_event(Realm::Event e) { this->event = e; }

void SharedState<void>::wait() { this->event.wait(); }
} // namespace FlexFlow