#include "realm-execution/realm_context.h"
#include "realm-execution/realm_task_id_t.h"
#include "realm-execution/task_id_t.dtg.h"
#include "utils/exception.h"

namespace FlexFlow {

RealmContext::RealmContext() {}

RealmContext::~RealmContext() {
  if (!this->outstanding_events.empty()) {
    Realm::Event outstanding = this->merge_outstanding_events();
    outstanding.wait();
  }
}

Allocator &RealmContext::get_current_device_allocator() const {
  NOT_IMPLEMENTED();
}

device_handle_t const &RealmContext::get_current_device_handle() const {
  NOT_IMPLEMENTED();
}
device_id_t const &RealmContext::get_current_device_idx() const {
  NOT_IMPLEMENTED();
}

Realm::Event RealmContext::get_outstanding_events() {
  Realm::Event result = this->merge_outstanding_events();
  this->outstanding_events.push_back(result);
  return result;
}

Realm::Event RealmContext::merge_outstanding_events() {
  Realm::Event result = Realm::Event::merge_events(this->outstanding_events);
  this->outstanding_events.clear();
  return result;
}

} // namespace FlexFlow
