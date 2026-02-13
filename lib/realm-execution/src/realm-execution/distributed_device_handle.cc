#include "realm-execution/distributed_device_handle.h"
#include "realm-execution/device_specific_managed_per_device_ff_handle.h"
#include "realm-execution/tasks/impl/device_handle_init_task.h"
#include "task-spec/device_specific.h"

namespace FlexFlow {

DistributedDeviceHandle::DistributedDeviceHandle(
    std::unordered_map<Realm::Processor,
                       DeviceSpecificManagedPerDeviceFFHandle> const &handles)
    : handles(handles) {}

DeviceSpecificManagedPerDeviceFFHandle const &
    DistributedDeviceHandle::at(Realm::Processor processor) const {
  return this->handles.at(processor);
}

DistributedDeviceHandle
    create_distributed_device_handle(RealmContext &ctx,
                                     size_t workSpaceSize,
                                     bool allowTensorOpMathConversion,
                                     Realm::Event precondition) {
  std::unordered_map<Realm::Processor, DeviceSpecificManagedPerDeviceFFHandle>
      handles;

  // Allocate space for the result before launching any tasks
  Realm::Machine::ProcessorQuery pq(Realm::Machine::get_machine());
  for (Realm::Processor proc : pq) {
    handles.insert({proc,
                    make_device_specific_managed_handle(
                        ctx.get_current_device_idx(), std::nullopt)});
  }

  for (auto &[proc, handle] : handles) {
    spawn_device_handle_init_task(ctx,
                                  proc,
                                  workSpaceSize,
                                  allowTensorOpMathConversion,
                                  &handle,
                                  precondition);
  }

  ctx.get_outstanding_events().wait();

  return DistributedDeviceHandle{handles};
}

} // namespace FlexFlow
