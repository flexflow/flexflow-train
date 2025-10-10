#ifndef _FLEXFLOW_LIB_LOCAL_PCG_EXECUTION_INCLUDE_LOCAL_PCG_EXECUTION_MAPPED_PER_DEVICE_OP_STATES_GROUP_H
#define _FLEXFLOW_LIB_LOCAL_PCG_EXECUTION_INCLUDE_LOCAL_PCG_EXECUTION_MAPPED_PER_DEVICE_OP_STATES_GROUP_H

#include "compiler/mapped_operator_task_group.h"
#include "pcg/machine_space_coordinate.dtg.h"
#include "task-spec/device_specific_per_device_op_state.dtg.h"
#include "utils/bidict/bidict.h"
#include "compiler/mapped_task_signature_tensor_key.dtg.h"

namespace FlexFlow {

struct MappedPerDeviceOpStatesGroup {
  MappedPerDeviceOpStatesGroup() = delete;

  explicit MappedPerDeviceOpStatesGroup(bidict<MachineSpaceCoordinate, DeviceSpecificPerDeviceOpState> const &per_device_op_states);

  [[nodiscard]] bool operator==(MappedPerDeviceOpStatesGroup const &) const;
  [[nodiscard]] bool operator!=(MappedPerDeviceOpStatesGroup const &) const;

  [[nodiscard]] bidict<MachineSpaceCoordinate, DeviceSpecificPerDeviceOpState> const &get_per_device_op_states() const;

private:
  bidict<MachineSpaceCoordinate, DeviceSpecificPerDeviceOpState> shard_bindings;

private:
  [[nodiscard]] std::tuple<
    decltype(shard_bindings) const &
  > tie() const;

  friend struct ::std::hash<MappedPerDeviceOpStatesGroup>;
};

std::string format_as(::FlexFlow::MappedPerDeviceOpStatesGroup const &);
std::ostream &operator<<(std::ostream &, ::FlexFlow::MappedPerDeviceOpStatesGroup const &);

} // namespace FlexFlow

namespace std {

template <>
struct hash<::FlexFlow::MappedPerDeviceOpStatesGroup> {
  size_t operator()(::FlexFlow::MappedPerDeviceOpStatesGroup const &) const;
};

} // namespace std
#endif
