#include "realm-execution/realm_allocator.h"
#include "kernels/device.h"
#include "pcg/device_type.dtg.h"

namespace FlexFlow {

RealmAllocator::RealmAllocator(Realm::Processor processor, Realm::Memory memory)
    : processor(processor), memory(memory) {}

void *RealmAllocator::allocate(size_t requested_memory_size) {
  Realm::Rect<1> bounds{Realm::Point<1>::ZEROES(),
                        Realm::Point<1>{requested_memory_size} -
                            Realm::Point<1>::ONES()};
  std::vector<size_t> field_sizes{1};
  Realm::RegionInstance inst;
  Realm::Event ready =
      Realm::RegionInstance::create_instance(inst,
                                             this->memory,
                                             bounds,
                                             field_sizes,
                                             0 /*SOA*/,
                                             Realm::ProfilingRequestSet{});
  ready.wait();
  void *ptr =
      inst.pointer_untyped(/*offset=*/0, /*datalen=*/requested_memory_size);
  ASSERT(ptr);
  this->ptr_instances.insert({ptr, inst});
  return ptr;
}

void RealmAllocator::deallocate(void *ptr) {
  this->ptr_instances.at(ptr).destroy(Realm::Event::NO_EVENT);
  this->ptr_instances.erase(ptr);
}

DeviceType RealmAllocator::get_allocation_device_type() const {
  switch (this->processor.kind()) {
    case Realm::Processor::Kind::LOC_PROC:
      return DeviceType::CPU;
    case Realm::Processor::Kind::TOC_PROC:
      return DeviceType::GPU;
    default:
      PANIC("Unhandled FwbTensorType", this->processor.kind());
  }
}

Allocator get_realm_allocator(Realm::Processor processor,
                              Realm::Memory memory) {
  Allocator allocator = Allocator::create<RealmAllocator>(processor, memory);
  return allocator;
}

} // namespace FlexFlow
