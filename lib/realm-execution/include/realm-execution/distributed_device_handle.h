#ifndef _FLEXFLOW_LIB_REALM_EXECUTION_INCLUDE_REALM_EXECUTION_DISTRIBUTED_DEVICE_HANDLE_H
#define _FLEXFLOW_LIB_REALM_EXECUTION_INCLUDE_REALM_EXECUTION_DISTRIBUTED_DEVICE_HANDLE_H

#include "realm-execution/device_specific_managed_per_device_ff_handle.dtg.h"
#include "realm-execution/hash/processor.h"
#include "realm-execution/realm.h"
#include "realm-execution/realm_context.h"
#include <unordered_map>

namespace FlexFlow {

struct DistributedDeviceHandle {
public:
  DistributedDeviceHandle() = delete;
  explicit DistributedDeviceHandle(
      std::unordered_map<Realm::Processor, DeviceSpecificManagedPerDeviceFFHandle> const
          &handles);

  DeviceSpecificManagedPerDeviceFFHandle const &
      at(Realm::Processor processor) const;

private:
  std::unordered_map<Realm::Processor, DeviceSpecificManagedPerDeviceFFHandle> handles;
};

DistributedDeviceHandle create_distributed_device_handle(
    RealmContext &ctx,
    size_t workSpaceSize,
    bool allowTensorOpMathConversion,
    Realm::Event precondition = Realm::Event::NO_EVENT);

} // namespace FlexFlow

#endif
