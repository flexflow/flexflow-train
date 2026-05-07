#include "realm-execution/realm_context.h"
#include "kernels/device_handle_t.dtg.h"
#include "kernels/device_handle_t.h"
#include "op-attrs/datatype.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "op-attrs/tensor_dims.dtg.h"
#include "pcg/device_id_t.h"
#include "pcg/device_type.dtg.h"
#include "realm-execution/realm_allocator.h"
#include "realm-execution/tasks/task_id_t.dtg.h"
#include "realm-execution/tasks/task_id_t.h"
#include "utils/containers/contains_key.h"
#include "utils/containers/transform.h"
#include "utils/exception.h"
#include "utils/nonnegative_int/nonnegative_int.h"
#include "utils/one_to_many/one_to_many.h"
#include "utils/positive_int/positive_int.h"
#include <realm/indexspace.h>
#include <realm/inst_layout.h>

namespace FlexFlow {
template <int N, typename T = int>
static Realm::Rect<N, T>
    rect_from_dims_with_offset(TensorDims const &dims,
                               std::vector<int> const &offsets) {
  std::vector<int> values;
  for (positive_int const &v : dims.ff_ordered) {
    values.push_back(v.int_from_positive_int());
  }
  ASSERT((int)values.size() == N);
  ASSERT((int)offsets.size() == N);

  std::vector<int> lo(N), hi(N);
  for (int i = 0; i < N; i++) {
    lo[i] = offsets[i];
    hi[i] = offsets[i] + values[i] - 1;
  }
  return Realm::Rect<N, T>{Realm::Point<N, T>{lo.data()},
                           Realm::Point<N, T>{hi.data()}};
}

template <int N>
static void make_row_major_dim_order(int (&dim_order)[N]) {
  for (int i = 0; i < N; i++) {
    dim_order[i] = i;
  }
}

RealmContext::RealmContext(Realm::Processor processor)
    : processor(processor),
      allocator(get_realm_allocator(
          processor, RealmContext::get_nearest_memory(processor))) {}

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

  // FIXME: find a more efficient way to implement this than scanning the
  // machine every time
  Realm::Machine::ProcessorQuery pq(Realm::Machine::get_machine());
  pq.same_address_space_as(proc);
  nonnegative_int idx{0};
  for (Realm::Processor p : pq) {
    if (p == proc) {
      break;
    }
    idx++;
  }

  switch (proc.kind()) {
    case Realm::Processor::LOC_PROC:
      return make_device_id_t_from_idx(idx, DeviceType::CPU);
    case Realm::Processor::TOC_PROC:
      return make_device_id_t_from_idx(idx, DeviceType::GPU);
    default:
      PANIC("Unhandled Realm::ProcessorKind", fmt::to_string(int{proc.kind()}));
  }
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
                             int priority,
                             std::optional<Realm::ReductionOpID> redop_id,
                             bool exclusive,
                             CopyDomain domain) {
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

  // set reduction op on dst field if provided
  if (redop_id.has_value()) {
    dst_field.set_redop(redop_id.value(), /*is_fold=*/false, exclusive);
  }

  // select which instance's index space to use as copy domain
  Realm::RegionInstance const domain_inst =
      (domain == CopyDomain::DST) ? dst_inst : src_inst;

  Realm::Event result;
  switch (src_piece_shape.dims.ff_ordered.num_dims()) {
#if REALM_MAX_DIM >= 1
    case 1:
      result = domain_inst.get_indexspace<1, int>().copy(
          {src_field}, {dst_field}, requests, wait_on, priority);
      break;
#endif
#if REALM_MAX_DIM >= 2
    case 2:
      result = domain_inst.get_indexspace<2, int>().copy(
          {src_field}, {dst_field}, requests, wait_on, priority);
      break;
#endif
#if REALM_MAX_DIM >= 3
    case 3:
      result = domain_inst.get_indexspace<3, int>().copy(
          {src_field}, {dst_field}, requests, wait_on, priority);
      break;
#endif
#if REALM_MAX_DIM >= 4
    case 4:
      result = domain_inst.get_indexspace<4, int>().copy(
          {src_field}, {dst_field}, requests, wait_on, priority);
      break;
#endif
#if REALM_MAX_DIM >= 5
    case 5:
      result = domain_inst.get_indexspace<5, int>().copy(
          {src_field}, {dst_field}, requests, wait_on, priority);
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

std::pair<Realm::RegionInstance, Realm::Event>
    RealmContext::create_instance_with_offset(
        Realm::Memory memory,
        TensorShape const &shape,
        std::vector<int> const &offsets,
        Realm::ProfilingRequestSet const &prs,
        Realm::Event wait_on) {
  std::vector<size_t> field_sizes{static_cast<size_t>(
      size_of_datatype(shape.data_type).int_from_positive_int())};
  Realm::RegionInstance inst;
  Realm::Event ready;
  switch (shape.dims.ff_ordered.num_dims()) {
#if REALM_MAX_DIM >= 1
    case 1:
      ready = Realm::RegionInstance::create_instance(
          inst,
          memory,
          rect_from_dims_with_offset<1>(shape.dims, offsets),
          field_sizes,
          0 /*SOA*/,
          prs,
          wait_on);
      break;
#endif
#if REALM_MAX_DIM >= 2
    case 2:
      ready = Realm::RegionInstance::create_instance(
          inst,
          memory,
          rect_from_dims_with_offset<2>(shape.dims, offsets),
          field_sizes,
          0 /*SOA*/,
          prs,
          wait_on);
      break;
#endif
#if REALM_MAX_DIM >= 3
    case 3:
      ready = Realm::RegionInstance::create_instance(
          inst,
          memory,
          rect_from_dims_with_offset<3>(shape.dims, offsets),
          field_sizes,
          0 /*SOA*/,
          prs,
          wait_on);
      break;
#endif
#if REALM_MAX_DIM >= 4
    case 4:
      ready = Realm::RegionInstance::create_instance(
          inst,
          memory,
          rect_from_dims_with_offset<4>(shape.dims, offsets),
          field_sizes,
          0 /*SOA*/,
          prs,
          wait_on);
      break;
#endif
#if REALM_MAX_DIM >= 5
    case 5:
      ready = Realm::RegionInstance::create_instance(
          inst,
          memory,
          rect_from_dims_with_offset<5>(shape.dims, offsets),
          field_sizes,
          0 /*SOA*/,
          prs,
          wait_on);
      break;
#endif
    default:
      PANIC("TensorShape dims greater than REALM_MAX_DIM: {}",
            shape.dims.ff_ordered.num_dims());
  }
  this->outstanding_events.push_back(ready);
  return {inst, ready};
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
std::pair<Realm::RegionInstance, Realm::Event>
    RealmContext::create_external_instance(
        Realm::Memory memory,
        TensorShape const &shape,
        std::vector<int> const &offsets,
        void *ptr,
        Realm::ProfilingRequestSet const &prs,
        Realm::Event wait_on) {

  std::vector<size_t> field_sizes{static_cast<size_t>(
      size_of_datatype(shape.data_type).int_from_positive_int())};
  Realm::InstanceLayoutConstraints ilc(field_sizes, /*block_size=*/0);

  Realm::RegionInstance inst;
  Realm::Event ready;

  switch (shape.dims.ff_ordered.num_dims()) {
#if REALM_MAX_DIM >= 1
    case 1: {
      int dim_order[1];
      make_row_major_dim_order(dim_order);
      Realm::Rect<1, int> rect =
          rect_from_dims_with_offset<1>(shape.dims, offsets);
      Realm::InstanceLayoutGeneric *layout =
          Realm::InstanceLayoutGeneric::choose_instance_layout<1, int>(
              Realm::IndexSpace<1, int>{rect}, ilc, dim_order);
      ready = Realm::RegionInstance::create_external(
          inst, memory, reinterpret_cast<uintptr_t>(ptr), layout, prs, wait_on);
      break;
    }
#endif
#if REALM_MAX_DIM >= 2
    case 2: {
      int dim_order[2];
      make_row_major_dim_order(dim_order);
      Realm::Rect<2, int> rect =
          rect_from_dims_with_offset<2>(shape.dims, offsets);
      Realm::InstanceLayoutGeneric *layout =
          Realm::InstanceLayoutGeneric::choose_instance_layout<2, int>(
              Realm::IndexSpace<2, int>{rect}, ilc, dim_order);
      ready = Realm::RegionInstance::create_external(
          inst, memory, reinterpret_cast<uintptr_t>(ptr), layout, prs, wait_on);
      break;
    }
#endif
#if REALM_MAX_DIM >= 3
    case 3: {
      int dim_order[3];
      make_row_major_dim_order(dim_order);
      Realm::Rect<3, int> rect =
          rect_from_dims_with_offset<3>(shape.dims, offsets);
      Realm::InstanceLayoutGeneric *layout =
          Realm::InstanceLayoutGeneric::choose_instance_layout<3, int>(
              Realm::IndexSpace<3, int>{rect}, ilc, dim_order);
      ready = Realm::RegionInstance::create_external(
          inst, memory, reinterpret_cast<uintptr_t>(ptr), layout, prs, wait_on);
      break;
    }
#endif
#if REALM_MAX_DIM >= 4
    case 4: {
      int dim_order[4];
      make_row_major_dim_order(dim_order);
      Realm::Rect<4, int> rect =
          rect_from_dims_with_offset<4>(shape.dims, offsets);
      Realm::InstanceLayoutGeneric *layout =
          Realm::InstanceLayoutGeneric::choose_instance_layout<4, int>(
              Realm::IndexSpace<4, int>{rect}, ilc, dim_order);
      ready = Realm::RegionInstance::create_external(
          inst, memory, reinterpret_cast<uintptr_t>(ptr), layout, prs, wait_on);
      break;
    }
#endif
#if REALM_MAX_DIM >= 5
    case 5: {
      int dim_order[5];
      make_row_major_dim_order(dim_order);
      Realm::Rect<5, int> rect =
          rect_from_dims_with_offset<5>(shape.dims, offsets);
      Realm::InstanceLayoutGeneric *layout =
          Realm::InstanceLayoutGeneric::choose_instance_layout<5, int>(
              Realm::IndexSpace<5, int>{rect}, ilc, dim_order);
      ready = Realm::RegionInstance::create_external(
          inst, memory, reinterpret_cast<uintptr_t>(ptr), layout, prs, wait_on);
      break;
    }
#endif
    default:
      PANIC("TensorShape dims greater than REALM_MAX_DIM: {}",
            shape.dims.ff_ordered.num_dims());
  }

  this->outstanding_events.push_back(ready);
  return {inst, ready};
}

Realm::Runtime RealmContext::get_runtime() {
  return this->runtime;
}

} // namespace FlexFlow
