#ifndef _FLEXFLOW_LIB_REALM_EXECUTION_INCLUDE_REALM_EXECUTION_DISTRIBUTED_DEVICE_HANDLE_H
#define _FLEXFLOW_LIB_REALM_EXECUTION_INCLUDE_REALM_EXECUTION_DISTRIBUTED_DEVICE_HANDLE_H

#include "kernels/managed_per_device_ff_handle.h"
#include "realm-execution/realm.h"
#include "realm-execution/realm_context.h"
#include "task-spec/device_specific.h"
#include <map>
#include <optional>

namespace FlexFlow {

struct DistributedDeviceHandle {
public:
  DistributedDeviceHandle() = delete;
  explicit DistributedDeviceHandle(
      std::map<Realm::Processor,
               DeviceSpecific<std::optional<ManagedPerDeviceFFHandle *>>> const
          &handles);

  DeviceSpecific<std::optional<ManagedPerDeviceFFHandle *>> const &
      at(Realm::Processor processor) const;

private:
  std::map<Realm::Processor,
           DeviceSpecific<std::optional<ManagedPerDeviceFFHandle *>>>
      handles;
};

DistributedDeviceHandle create_distributed_device_handle(
    RealmContext &ctx,
    size_t workSpaceSize,
    bool allowTensorOpMathConversion,
    Realm::Event precondition = Realm::Event::NO_EVENT);

} // namespace FlexFlow

#endif
