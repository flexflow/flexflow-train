#ifndef _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_INIT_OP_TASK_IMPL_FUNCTION_H
#define _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_INIT_OP_TASK_IMPL_FUNCTION_H

#include "task-spec/device_specific_device_states.dtg.h"
#include "task-spec/task_argument_accessor.h"

namespace FlexFlow {

struct InitOpTaskImplFunction {

  DeviceSpecificDeviceStates (*function_ptr)(TaskArgumentAccessor const &);

  bool operator==(InitOpTaskImplFunction const &) const;
  bool operator!=(InitOpTaskImplFunction const &) const;
  bool operator<(InitOpTaskImplFunction const &) const;
  bool operator>(InitOpTaskImplFunction const &) const;
  bool operator<=(InitOpTaskImplFunction const &) const;
  bool operator>=(InitOpTaskImplFunction const &) const;
};

std::string format_as(InitOpTaskImplFunction const &x);
std::ostream &operator<<(std::ostream &s, InitOpTaskImplFunction const &x);

} // namespace FlexFlow

namespace std {
template <>
struct hash<::FlexFlow::InitOpTaskImplFunction> {
  size_t operator()(::FlexFlow::InitOpTaskImplFunction const &) const;
};
} // namespace std

#endif
