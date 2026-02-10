#include "realm-execution/realm_context.h"
#include "op-attrs/datatype.h"
#include "op-attrs/tensor_dims.dtg.h"
#include "pcg/device_type.dtg.h"
#include "realm-execution/tasks/realm_task_id_t.h"
#include "realm-execution/tasks/task_id_t.dtg.h"
#include "utils/containers/contains_key.h"
#include "utils/containers/transform.h"
#include "utils/exception.h"
#include "utils/nonnegative_int/nonnegative_int.h"
#include "utils/one_to_many/one_to_many.h"
#include "utils/positive_int/positive_int.h"

namespace FlexFlow {

RealmContext::RealmContext(Realm::Processor proc) : processor(proc) {}

RealmContext::~RealmContext() {
  if (!this->outstanding_events.empty()) {
    Realm::Event outstanding = this->merge_outstanding_events();
    outstanding.wait();
  }
}

static std::tuple<Realm::AddressSpace, Realm::Processor::Kind, nonnegative_int>
    convert_machine_space_coordinate(
        MachineSpaceCoordinate const &device_coord) {
  Realm::AddressSpace as = int{device_coord.node_idx};
  Realm::Processor::Kind kind;
  switch (device_coord.device_type) {
    case DeviceType::CPU:
      kind = Realm::Processor::Kind::LOC_PROC;
      break;
    case DeviceType::GPU:
      kind = Realm::Processor::Kind::TOC_PROC;
      break;
    default:
      PANIC("Unhandled DeviceType", fmt::to_string(device_coord.device_type));
      break;
  }
  nonnegative_int proc_in_node = device_coord.device_idx;
  return std::tuple{as, kind, proc_in_node};
}

Realm::Processor RealmContext::map_device_coord_to_processor(
    MachineSpaceCoordinate const &device_coord) {
  this->discover_machine_topology();
  auto [as, kind, proc_in_node] =
      convert_machine_space_coordinate(device_coord);
  return this->processors.at(std::pair{as, kind}).at(int{proc_in_node});
}

Realm::Memory RealmContext::get_nearest_memory(Realm::Processor proc) const {
  // FIMXE: this isn't going to do what you expect until
  // https://github.com/StanfordLegion/realm/pull/392 merges
  Realm::Machine::MemoryQuery mq(Realm::Machine::get_machine());
  mq.best_affinity_to(proc);
  ASSERT(mq.count() > 0);
  return mq.first();
}

Realm::Processor RealmContext::get_current_processor() const {
  return this->processor;
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

Realm::Event
    RealmContext::spawn_task(Realm::Processor proc,
                             task_id_t task_id,
                             void const *args,
                             size_t arglen,
                             Realm::ProfilingRequestSet const &requests,
                             Realm::Event wait_on,
                             int priority) {
  Realm::Event result = proc.spawn(get_realm_task_id_for_task_id(task_id),
                                   args,
                                   arglen,
                                   requests,
                                   wait_on,
                                   priority);
  this->outstanding_events.push_back(result);
  return result;
}

Realm::Event RealmContext::collective_spawn_task(Realm::Processor target_proc,
                                                 task_id_t task_id,
                                                 void const *args,
                                                 size_t arglen,
                                                 Realm::Event wait_on,
                                                 int priority) {
  Realm::Event result =
      this->runtime.collective_spawn(target_proc,
                                     get_realm_task_id_for_task_id(task_id),
                                     args,
                                     arglen,
                                     wait_on,
                                     priority);
  this->outstanding_events.push_back(result);
  return result;
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

void RealmContext::discover_machine_topology() {
  if (!this->processors.empty()) {
    return;
  }

  Realm::Machine::ProcessorQuery pq(Realm::Machine::get_machine());
  for (Realm::Processor proc : pq) {
    Realm::AddressSpace as = proc.address_space();
    Realm::Processor::Kind kind = proc.kind();
    this->processors[std::pair{as, kind}].push_back(proc);
  }
}

} // namespace FlexFlow
