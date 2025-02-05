#ifndef _FLEXFLOW_LOCAL_EXECUTION_TASK_RESULT_H
#define _FLEXFLOW_LOCAL_EXECUTION_TASK_RESULT_H

#include "realm-backend/driver.h"
#include <cassert>

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
  Realm::RegionInstance inst;

  SharedState() = delete;
  SharedState(Realm::Memory);
  void set_event(Realm::Event);
  void set_value(T &&);
  void wait();
  T get_value();
};

// Specialization of SharedState for the `void` type, as it does not carry a
// value.
template <> struct SharedState<void> {
  // synchronization primitives
  Realm::Event event = Realm::Event::NO_EVENT;

  SharedState() = default;
  void set_event(Realm::Event);
  void wait();
};

/**
 * @brief Future class template that allows retrieving the result from a
 * SharedState object. It is used to access the value once the Promise has been
 * fulfilled, and provides mechanisms to block the current thread until the
 * result is available.
 */
template <typename T> class Future {
public:
  explicit Future(std::shared_ptr<SharedState<T>> state)
      : state_(std::move(state)) {}
  explicit Future(T value) : value_(std::move(value)) {}
  void set_event(Realm::Event e) { state_->set_event(e); }
  T get() {
    value_ = state_->get_value();
    return value_;
  }
  void wait() { state_->wait(); }

private:
  std::shared_ptr<SharedState<T>> state_;
  T value_;
};

// Specialization of Future for the `void` type, as it does not carry a value.
template <> class Future<void> {
public:
  explicit Future(std::shared_ptr<SharedState<void>> state)
      : state_(std::move(state)) {}
  explicit Future() = default;
  void set_event(Realm::Event e) { state_->set_event(e); }
  void wait() { state_->wait(); }

private:
  std::shared_ptr<SharedState<void>> state_;
};

/**
 * @brief Promise class template that allows setting a result in a SharedState
 * object. It is used to fulfill a Future with a value, and provides methods to
 * notify the waiting Future of completion.
 */
template <typename T> class Promise {
public:
  Promise() = delete;
  Promise(Realm::Memory mem) : state_(std::make_shared<SharedState<T>>(mem)) {}
  Future<T> get_future() { return Future<T>(state_); }
  void set_value(T &&value) const { state_->set_value(std::move(value)); }

private:
  std::shared_ptr<SharedState<T>> state_;
};

// Specialization of Promise for the `void` type, as it does not carry a value.
template <> class Promise<void> {
public:
  Promise() : state_(std::make_shared<SharedState<void>>()) {}
  Future<void> get_future() { return Future<void>(state_); }

private:
  std::shared_ptr<SharedState<void>> state_;
};

} // namespace FlexFlow

#endif