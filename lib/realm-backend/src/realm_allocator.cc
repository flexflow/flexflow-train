#include "realm-backend/realm_allocator.h"
#include "utils/containers/contains_key.h"

namespace FlexFlow {

using namespace Realm;

/*********** RealmAllocatorImpl ***********/

RealmAllocatorImpl::RealmAllocatorImpl(Processor proc) : proc(proc) {
  mem = Machine::MemoryQuery(Machine::get_machine())
            .only_kind(Memory::GPU_FB_MEM)
            .best_affinity_to(proc)
            .first();
}

// TODO: now the region instance only corresponds to one tensor
void *RealmAllocatorImpl::allocate(size_t requested_memory_size) {
  Rect<1> bounds(Point<1>(0), Point<1>(requested_memory_size - 1));
  RegionInstance requested_instance = RegionInstance::NO_INST;
  RegionInstance::create_instance(requested_instance, mem, bounds, field_sizes,
                                  /*SOA*/ 1, ProfilingRequestSet())
      .wait();
  void *ptr = requested_instance.pointer_untyped(0, 0);
  this->ptrs.insert({ptr, requested_instance});
  return ptr;
}

void RealmAllocatorImpl::deallocate(void *ptr) {
  if (this->ptrs.count(ptr)) {
    RegionInstance region = this->ptrs.at(ptr);
    region.destroy();
  } else {
    throw std::runtime_error(
        "Deallocating a pointer that was not allocated by this Allocator");
  }
}

DeviceType RealmAllocatorImpl::get_allocation_device_type() const {
  return DeviceType::GPU;
}

Allocator create_realm_memory_allocator(Processor proc) {
  return Allocator::create<RealmAllocatorImpl>(proc);
}

} // namespace FlexFlow
