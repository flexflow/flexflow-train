#include "realm-execution/realm_context.h"
#include "op-attrs/datatype.h"
#include "realm-execution/realm_task_id_t.h"
#include "realm-execution/task_id_t.dtg.h"
#include "utils/exception.h"
#include "utils/positive_int/positive_int.h"

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

std::pair<Realm::RegionInstance, Realm::Event>
    RealmContext::create_instance(Realm::Memory memory,
                                  TensorShape const &shape,
                                  Realm::ProfilingRequestSet const &prs,
                                  Realm::Event wait_on) {
  std::vector<int> dims{shape.dims.ff_ordered.begin(),
                        shape.dims.ff_ordered.end()};
  std::vector<size_t> field_sizes{
      static_cast<size_t>(int{size_of_datatype(shape.data_type)})};
  Realm::RegionInstance inst;
  Realm::Event ready;
  switch (shape.dims.ff_ordered.num_dims()) {
#if REALM_MAX_DIM >= 1
    case 1:
      ready = Realm::RegionInstance::create_instance(
          inst,
          memory,
          Realm::Rect<1>(Realm::Point<1>::ZEROES(),
                         Realm::Point<1>(dims.data()) -
                             Realm::Point<1>::ONES()),
          field_sizes,
          /*block_size=*/0 /*SOA*/,
          prs,
          wait_on);
      break;
#endif
#if REALM_MAX_DIM >= 2
    case 2:
      ready = Realm::RegionInstance::create_instance(
          inst,
          memory,
          Realm::Rect<2>(Realm::Point<2>::ZEROES(),
                         Realm::Point<2>(dims.data()) -
                             Realm::Point<2>::ONES()),
          field_sizes,
          /*block_size=*/0 /*SOA*/,
          prs,
          wait_on);
      break;
#endif
#if REALM_MAX_DIM >= 3
    case 3:
      ready = Realm::RegionInstance::create_instance(
          inst,
          memory,
          Realm::Rect<3>(Realm::Point<3>::ZEROES(),
                         Realm::Point<3>(dims.data()) -
                             Realm::Point<3>::ONES()),
          field_sizes,
          /*block_size=*/0 /*SOA*/,
          prs,
          wait_on);
      break;
#endif
#if REALM_MAX_DIM >= 4
    case 4:
      ready = Realm::RegionInstance::create_instance(
          inst,
          memory,
          Realm::Rect<4>(Realm::Point<4>::ZEROES(),
                         Realm::Point<4>(dims.data()) -
                             Realm::Point<4>::ONES()),
          field_sizes,
          /*block_size=*/0 /*SOA*/,
          prs,
          wait_on);
      break;
#endif
#if REALM_MAX_DIM >= 5
    case 5:
      ready = Realm::RegionInstance::create_instance(
          inst,
          memory,
          Realm::Rect<5>(Realm::Point<5>::ZEROES(),
                         Realm::Point<5>(dims.data()) -
                             Realm::Point<5>::ONES()),
          field_sizes,
          /*block_size=*/0 /*SOA*/,
          prs,
          wait_on);
      break;
#endif
    default:
      PANIC("TensorShape dims greater than REALM_MAX_DIM",
            fmt::to_string(shape.dims.ff_ordered.num_dims()));
      break;
  }
  this->outstanding_events.push_back(ready);
  return std::pair{inst, ready};
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
