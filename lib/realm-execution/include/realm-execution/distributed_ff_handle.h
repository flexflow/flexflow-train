#ifndef _FLEXFLOW_LIB_REALM_EXECUTION_INCLUDE_REALM_EXECUTION_DISTRIBUTED_DEVICE_HANDLE_H
#define _FLEXFLOW_LIB_REALM_EXECUTION_INCLUDE_REALM_EXECUTION_DISTRIBUTED_DEVICE_HANDLE_H

#include "realm-execution/device_specific_managed_per_device_ff_handle.h"
#include "realm-execution/realm.h"
#include "realm-execution/realm_context.h"
#include <unordered_map>

namespace FlexFlow {

/**
 * \brief Tracks the \ref device_handle_t (i.e., FFHandle) for each %GPU, both
 * local and remote. A GPU here is represented by a Realm::Processor.
 */
struct DistributedFfHandle {
public:
  DistributedFfHandle() = delete;
  explicit DistributedFfHandle(
      std::unordered_map<Realm::Processor,
                         DeviceSpecificManagedPerDeviceFFHandle> const
          &handles);

  DeviceSpecificManagedPerDeviceFFHandle const &
      at(Realm::Processor processor) const;

private:
  std::unordered_map<Realm::Processor, DeviceSpecificManagedPerDeviceFFHandle>
      handles;
};

/**
 * \brief Launches tasks (using \ref spawn_ff_handle_init_task) to create
 * the \ref device_handle_t ""s for each %GPU and packages the results into a
 * DistributedFfHandle.
 *
 * \relates DistributedFfHandle
 */
DistributedFfHandle create_distributed_ff_handle(
    RealmContext &ctx,
    size_t workSpaceSize,
    bool allowTensorOpMathConversion,
    Realm::Event precondition = Realm::Event::NO_EVENT);

} // namespace FlexFlow

#endif
