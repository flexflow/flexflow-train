#ifndef _FLEXFLOW_LIB_REALM_EXECUTION_INCLUDE_REALM_EXECUTION_REALM_ALLOCATOR_H
#define _FLEXFLOW_LIB_REALM_EXECUTION_INCLUDE_REALM_EXECUTION_REALM_ALLOCATOR_H

#include "kernels/allocation.h"
#include "realm-execution/realm.h"

namespace FlexFlow {

struct RealmAllocator : public IAllocator {
  RealmAllocator(Realm::Processor processor, Realm::Memory memory);

  RealmAllocator() = delete;
  RealmAllocator(RealmAllocator const &) = delete;
  RealmAllocator(RealmAllocator &&) = delete;
  ~RealmAllocator() = default;

  void *allocate(size_t) override;
  void deallocate(void *) override;

  DeviceType get_allocation_device_type() const override;

private:
  Realm::Processor processor;
  Realm::Memory memory;
  std::unordered_map<void *, Realm::RegionInstance> ptr_instances;
};
CHECK_RC_COPY_VIRTUAL_COMPLIANT(RealmAllocator);

Allocator get_realm_allocator(Realm::Processor processor, Realm::Memory memory);

} // namespace FlexFlow

#endif
