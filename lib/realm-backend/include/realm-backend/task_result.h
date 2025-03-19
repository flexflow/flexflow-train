#ifndef _FLEXFLOW_LOCAL_EXECUTION_TASK_RESULT_H
#define _FLEXFLOW_LOCAL_EXECUTION_TASK_RESULT_H

#include "realm-backend/driver.h"
#include <cassert>
#include <optional>

namespace FlexFlow {

/**
 * @brief SharedState class template that holds the state for both the Promise
 * and Future objects. It is responsible for storing the result value and
 * synchronization between the producer (Promise) and consumer (Future).
 */
template <typename T> struct SharedState {
  // synchronization primitives
  Realm::Event event = Realm::Event::NO_EVENT;
  // where the result is stored
  Realm::RegionInstance inst = Realm::RegionInstance::NO_INST;

  SharedState() = default;
  SharedState(Realm::Memory mem) {
    Realm::Rect<1> bounds(Realm::Point<1>(0), Realm::Point<1>(0));
    Realm::RegionInstance::create_instance(
        this->inst, mem, bounds, {sizeof(T)}, /*SOA*/ 1,
        Realm::ProfilingRequestSet(), Realm::Event::NO_EVENT)
        .wait();
  }
  void set_event(Realm::Event e) { this->event = e; }
  void set_value(T &&value) const {
    assert(this->inst.exists());
    Realm::GenericAccessor<T, 1> acc(this->inst, 0);
    acc[Realm::Point<1>(0)] = std::move(value);
  }
  void wait() { this->event.wait(); }
  T get_value() {
    wait();
    assert(this->inst.exists());
    Realm::GenericAccessor<T, 1> acc(this->inst, 0);
    T value = acc[Realm::Point<1>(0)];
    this->inst.destroy();
    return value;
  }
};

// Specialization of SharedState for the `void` type, as it does not carry a
// value.
template <> struct SharedState<void> {
  // synchronization primitives
  Realm::Event event = Realm::Event::NO_EVENT;

  SharedState() = default;
  void set_event(Realm::Event e) { this->event = e; }
  void wait() { this->event.wait(); }
};

/**
 * @brief Future class template that allows retrieving the result from a
 * SharedState object. It is used to access the value once the Promise has been
 * fulfilled, and provides mechanisms to block the current thread until the
 * result is available.
 */
template <typename T> class Future {
public:
  explicit Future(SharedState<T> state) : state_(state) {}
  explicit Future() = default;
  explicit Future(T value) : value_(std::move(value)) {}
  void set_event(Realm::Event e) { state_.set_event(e); }
  T get() {
    if (!value_.has_value()) {
      value_ = std::make_optional(state_.get_value());
    }
    return value_.value();
  }
  void wait() { state_.wait(); }

private:
  SharedState<T> state_;
  std::optional<T> value_;
};

// Specialization of Future for the `void` type, as it does not carry a value.
template <> class Future<void> {
public:
  explicit Future(SharedState<void> state) : state_(state) {}
  explicit Future() = default;
  void set_event(Realm::Event e) { state_.set_event(e); }
  void get() { state_.wait(); }
  void wait() { state_.wait(); }

private:
  SharedState<void> state_;
};

/**
 * @brief Promise class template that allows setting a result in a SharedState
 * object. It is used to fulfill a Future with a value, and provides methods to
 * notify the waiting Future of completion.
 */
template <typename T> class Promise {
public:
  Promise() = delete;
  Promise(Realm::Memory mem) : state_(SharedState<T>(mem)) {}
  Future<T> get_future() { return Future<T>(state_); }
  void set_value(T &&value) const { state_.set_value(std::move(value)); }

private:
  SharedState<T> state_;
};

// Specialization of Promise for the `void` type, as it does not carry a value.
template <> class Promise<void> {
public:
  Promise() : state_(SharedState<void>()) {}
  Future<void> get_future() { return Future<void>(state_); }

private:
  SharedState<void> state_;
};

} // namespace FlexFlow

#endif