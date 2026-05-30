#include "realm-execution/realm_context.h"
#include "kernels/device_handle_t.dtg.h"
#include "kernels/device_handle_t.h"
#include "op-attrs/datatype.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "op-attrs/tensor_dims.dtg.h"
#include "realm-execution/processor_kind.h"
#include "realm-execution/realm_allocator.h"
#include "realm-execution/tasks/task_id_t.dtg.h"
#include "realm-execution/tasks/task_id_t.h"
#include "utils/containers/contains_key.h"
#include "utils/containers/transform.h"
#include "utils/exception.h"
#include "utils/nonnegative_int/nonnegative_int.h"
#include "utils/one_to_many/one_to_many.h"
#include "utils/optional.h"
#include "utils/positive_int/positive_int.h"

namespace FlexFlow {

RealmContext::RealmContext(Realm::Processor processor)
    : processor(processor),
      allocator(get_realm_allocator(
          processor, RealmContext::get_nearest_memory(processor))) {
  if (processor != Realm::Processor::NO_PROC) {
    this->discover_machine_topology();
  }
}

RealmContext::~RealmContext() {
  if (!this->outstanding_events.empty()) {
    Realm::Event outstanding = this->merge_outstanding_events();
    outstanding.wait();
  }
}

static std::tuple<Realm::AddressSpace, Realm::Processor::Kind, nonnegative_int>
    convert_machine_space_coordinate(MachineSpaceCoordinate const &device_coord,
                                     DeviceType device_type) {
  Realm::AddressSpace as = int{device_coord.node_idx};
  Realm::Processor::Kind kind = processor_kind_from_device_type(device_type);
  nonnegative_int proc_in_node = device_coord.device_idx;
  return std::tuple{as, kind, proc_in_node};
}

Realm::Processor RealmContext::map_device_coord_to_processor(
    device_id_t const &device_id) const {
  return assert_unwrap(this->processors).at_r(device_id);
}

device_id_t
    RealmContext::map_processor_to_device_coord(Realm::Processor p) const {
  return assert_unwrap(this->processors).at_l(p);
}

Realm::Memory RealmContext::get_nearest_memory(Realm::Processor proc) {
  if (!proc.exists()) {
    return Realm::Memory::NO_MEMORY;
  }

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

Allocator &RealmContext::get_current_device_allocator() {
  return this->allocator;
}

device_id_t RealmContext::get_current_device_idx() const {
  Realm::Processor proc = this->get_current_processor();
  return this->map_processor_to_device_coord(proc);
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

template <int N, typename T = int>
static Realm::Rect<N, T> rect_from_dims(TensorDims const &dims) {
  std::vector<int> values{dims.ff_ordered.begin(), dims.ff_ordered.end()};
  ASSERT(values.size() == N);
  return Realm::Rect<N, T>{Realm::Point<N, T>::ZEROES(),
                           Realm::Point<N, T>{values.data()} -
                               Realm::Point<N, T>::ONES()};
}

template <int N, typename T = int>
static Realm::IndexSpace<N, T> ispace_from_dims(TensorDims const &dims) {
  Realm::Rect<N, T> rect = rect_from_dims<N, T>(dims);
  return Realm::IndexSpace<N, T>{rect};
}

Realm::Event
    RealmContext::issue_copy(ParallelTensorShape const &src_shape,
                             Realm::RegionInstance src_inst,
                             ParallelTensorShape const &dst_shape,
                             Realm::RegionInstance dst_inst,
                             Realm::ProfilingRequestSet const &requests,
                             Realm::Event wait_on,
                             int priority) {
  TensorShape src_piece_shape = get_piece_shape(src_shape);
  TensorShape dst_piece_shape = get_piece_shape(dst_shape);
  ASSERT(src_piece_shape == dst_piece_shape); // For now, assume they match

  Realm::CopySrcDstField src_field;
  src_field.set_field(
      /*inst=*/src_inst,
      /*field_id=*/0,
      /*size=*/
      static_cast<size_t>(
          size_of_datatype(src_piece_shape.data_type).int_from_positive_int()),
      /*subfield_offset=*/0);
  Realm::CopySrcDstField dst_field;
  dst_field.set_field(
      /*inst=*/dst_inst,
      /*field_id=*/0,
      /*size=*/
      static_cast<size_t>(
          size_of_datatype(src_piece_shape.data_type).int_from_positive_int()),
      /*subfield_offset=*/0);

  Realm::Event result;
  switch (src_piece_shape.dims.ff_ordered.num_dims()) {
#if REALM_MAX_DIM >= 1
    case 1:
      result = ispace_from_dims<1>(src_piece_shape.dims)
                   .copy({src_field}, {dst_field}, requests, wait_on, priority);
      break;
#endif
#if REALM_MAX_DIM >= 2
    case 2:
      result = ispace_from_dims<2>(src_piece_shape.dims)
                   .copy({src_field}, {dst_field}, requests, wait_on, priority);
      break;
#endif
#if REALM_MAX_DIM >= 3
    case 3:
      result = ispace_from_dims<3>(src_piece_shape.dims)
                   .copy({src_field}, {dst_field}, requests, wait_on, priority);
      break;
#endif
#if REALM_MAX_DIM >= 4
    case 4:
      result = ispace_from_dims<4>(src_piece_shape.dims)
                   .copy({src_field}, {dst_field}, requests, wait_on, priority);
      break;
#endif
#if REALM_MAX_DIM >= 5
    case 5:
      result = ispace_from_dims<5>(src_piece_shape.dims)
                   .copy({src_field}, {dst_field}, requests, wait_on, priority);
      break;
#endif
    default:
      PANIC("TensorShape dims greater than REALM_MAX_DIM: {}",
            src_piece_shape.dims.ff_ordered.num_dims());
      break;
  }
  this->outstanding_events.push_back(result);
  return result;
}

std::pair<Realm::RegionInstance, Realm::Event>
    RealmContext::create_instance(Realm::Memory memory,
                                  TensorShape const &shape,
                                  Realm::ProfilingRequestSet const &prs,
                                  Realm::Event wait_on) {
  std::vector<size_t> field_sizes{static_cast<size_t>(
      size_of_datatype(shape.data_type).int_from_positive_int())};
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
  if (this->processors.has_value()) {
    return;
  }

  std::unordered_map<std::pair<nonnegative_int, DeviceType>, nonnegative_int>
      next_device_idx;

  auto fresh_device_id = [&](nonnegative_int node_idx,
                             DeviceType device_type) -> device_id_t {
    std::pair<nonnegative_int, DeviceType> key =
        std::pair{node_idx, device_type};
    if (!contains_key(next_device_idx, key)) {
      next_device_idx.insert({key, 0_n});
    }

    nonnegative_int device_idx = next_device_idx.at(key);
    next_device_idx.at(key)++;

    return device_id_t{
        MachineSpaceCoordinate{node_idx, device_idx},
        device_type,
    };
  };

  bidict<Realm::Processor, device_id_t> procs;
  Realm::Machine::ProcessorQuery pq(Realm::Machine::get_machine());
  for (Realm::Processor proc : pq) {
    Realm::AddressSpace as = proc.address_space();
    Realm::Processor::Kind kind = proc.kind();

    nonnegative_int node_idx = nonnegative_int{static_cast<int>(as)};

    if (kind != Realm::Processor::LOC_PROC &&
        kind != Realm::Processor::TOC_PROC) {
      continue;
    }

    DeviceType device_type = device_type_from_processor_kind(kind);
    device_id_t coord = fresh_device_id(node_idx, device_type);
    procs.equate_strict(proc, coord);
  }

  this->processors = procs;
}

Realm::Runtime RealmContext::get_runtime() {
  return this->runtime;
}

} // namespace FlexFlow
