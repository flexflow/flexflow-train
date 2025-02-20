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
RealmRegion RealmAllocatorImpl::allocate(size_t requested_memory_size) {
  Rect<1> bounds(Point<1>(0), Point<1>(requested_memory_size - 1));
  RegionInstance requested_instance = RegionInstance::NO_INST;
  RegionInstance::create_instance(requested_instance, mem, bounds, field_sizes,
                                  /*SOA*/ 1, ProfilingRequestSet())
      .wait();
  void *ptr = requested_instance.pointer_untyped(0, 0);
  this->ptrs.insert({requested_instance, ptr});
  return {requested_instance, this};
}

void RealmAllocatorImpl::deallocate(RealmRegion region) {
  if (region.allocator == this and contains_key(this->ptrs, region.instance)) {
    RegionInstance instance = this->ptrs.at(region.instance);
    instance.destroy();
  } else {
    throw std::runtime_error(
        "Deallocating a pointer that was not allocated by this Allocator");
  }
}


/*********** RealmAllocator ***********/

RealmRegion RealmAllocator::allocate(size_t mem_size) {
  return this->i_allocator->allocate(mem_size);
}

void RealmAllocator::deallocate(RealmRegion region) {
  this->i_allocator->deallocate(region);
}

RealmAllocator create_realm_memory_allocator(Processor proc) {
  return RealmAllocator::create<RealmAllocatorImpl>(proc);
}

} // namespace FlexFlow
