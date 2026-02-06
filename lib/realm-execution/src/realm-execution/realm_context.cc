#include "realm-execution/realm_context.h"
#include "op-attrs/datatype.h"
#include "op-attrs/tensor_dims.dtg.h"
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

template <int N>
static Realm::Rect<N> rect_from_dims(TensorDims const &dims) {
  std::vector<int> values{dims.ff_ordered.begin(), dims.ff_ordered.end()};
  return Realm::Rect<N>{Realm::Point<N>::ZEROES(),
                        Realm::Point<N>{values.data()} -
                            Realm::Point<N>::ONES()};
}

std::pair<Realm::RegionInstance, Realm::Event>
    RealmContext::create_instance(Realm::Memory memory,
                                  TensorShape const &shape,
                                  Realm::ProfilingRequestSet const &prs,
                                  Realm::Event wait_on) {
  std::vector<size_t> field_sizes{
      static_cast<size_t>(int{size_of_datatype(shape.data_type)})};
  Realm::RegionInstance inst;
  Realm::Event ready;
  switch (shape.dims.ff_ordered.num_dims()) {
#if REALM_MAX_DIM >= 1
    case 1:
      ready =
          Realm::RegionInstance::create_instance(inst,
                                                 memory,
                                                 rect_from_dims<1>(shape.dims),
                                                 field_sizes,
                                                 0 /*SOA*/,
                                                 prs,
                                                 wait_on);
      break;
#endif
#if REALM_MAX_DIM >= 2
    case 2:
      ready =
          Realm::RegionInstance::create_instance(inst,
                                                 memory,
                                                 rect_from_dims<2>(shape.dims),
                                                 field_sizes,
                                                 0 /*SOA*/,
                                                 prs,
                                                 wait_on);
      break;
#endif
#if REALM_MAX_DIM >= 3
    case 3:
      ready =
          Realm::RegionInstance::create_instance(inst,
                                                 memory,
                                                 rect_from_dims<3>(shape.dims),
                                                 field_sizes,
                                                 0 /*SOA*/,
                                                 prs,
                                                 wait_on);
      break;
#endif
#if REALM_MAX_DIM >= 4
    case 4:
      ready =
          Realm::RegionInstance::create_instance(inst,
                                                 memory,
                                                 rect_from_dims<4>(shape.dims),
                                                 field_sizes,
                                                 0 /*SOA*/,
                                                 prs,
                                                 wait_on);
      break;
#endif
#if REALM_MAX_DIM >= 5
    case 5:
      ready =
          Realm::RegionInstance::create_instance(inst,
                                                 memory,
                                                 rect_from_dims<5>(shape.dims),
                                                 field_sizes,
                                                 0 /*SOA*/,
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
