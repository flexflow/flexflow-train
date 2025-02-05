#ifndef _FLEXFLOW_REALM_BACKEND_REALM_ALLOCATOR_H
#define _FLEXFLOW_REALM_BACKEND_REALM_ALLOCATOR_H

#include "realm-backend/driver.h"
#include "realm.h"
#include <realm/event.h>

namespace FlexFlow {

struct RealmAllocatorImpl;

struct RealmRegion {
  Realm::RegionInstance instance;
  RealmAllocatorImpl *allocator;
};

struct RealmAllocatorImpl {
  RealmAllocatorImpl() = delete;
  RealmAllocatorImpl(RealmAllocatorImpl const &) = delete;
  RealmAllocatorImpl(RealmAllocatorImpl &&) = delete;
  RealmAllocatorImpl(Realm::Processor);
  ~RealmAllocatorImpl() = default;

  RealmRegion allocate(size_t);
  void deallocate(RealmRegion);

private:
  std::unordered_map<Realm::RegionInstance, void *> ptrs;
  Realm::Processor proc;
  Realm::Memory mem;
  std::vector<size_t> field_sizes = {sizeof(char)};
};

struct RealmAllocator {
  RealmAllocator() = delete;

  RealmRegion allocate(size_t);
  void deallocate(RealmRegion);

  template <typename T, typename... Args>
  static typename std::enable_if<std::is_base_of<RealmAllocatorImpl, T>::value,
                                 RealmAllocator>::type
  create(Args &&...args) {
    return RealmAllocator(std::make_shared<T>(std::forward<Args>(args)...));
  }

  RealmAllocator(std::shared_ptr<RealmAllocatorImpl> ptr) : i_allocator(ptr) {};
  RealmAllocator(RealmAllocator const &allocator)
      : i_allocator(allocator.i_allocator) {};

private:
  std::shared_ptr<RealmAllocatorImpl> i_allocator;
};

RealmAllocator create_realm_memory_allocator(Realm::Processor);

} // namespace FlexFlow

#endif