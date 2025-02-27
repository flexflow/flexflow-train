#ifndef _FLEXFLOW_REALM_BACKEND_REALM_ALLOCATOR_H
#define _FLEXFLOW_REALM_BACKEND_REALM_ALLOCATOR_H

#include "realm-backend/driver.h"
#include "realm.h"
#include "kernels/allocation.h"
#include <realm/event.h>

namespace FlexFlow {

struct RealmAllocatorImpl;

struct RealmAllocatorImpl : public IAllocator {
  RealmAllocatorImpl() = delete;
  RealmAllocatorImpl(RealmAllocatorImpl const &) = delete;
  RealmAllocatorImpl(RealmAllocatorImpl &&) = delete;
  RealmAllocatorImpl(Realm::Processor);
  ~RealmAllocatorImpl() = default;

  void *allocate(size_t) override;
  void deallocate(void *) override;

private:
  std::unordered_map<void *, Realm::RegionInstance> ptrs;
  Realm::Processor proc;
  Realm::Memory mem;
  std::vector<size_t> field_sizes = {sizeof(char)};
};

Allocator create_realm_memory_allocator(Realm::Processor);

} // namespace FlexFlow

#endif