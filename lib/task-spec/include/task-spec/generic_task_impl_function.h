#ifndef _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_GENERIC_TASK_IMPL_FUNCTION_H
#define _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_GENERIC_TASK_IMPL_FUNCTION_H

#include "task-spec/device_specific_per_device_op_state.dtg.h"
#include "task-spec/task_argument_accessor/task_argument_accessor.h"

namespace FlexFlow {

struct GenericTaskImplFunction {

  void (*function_ptr)(TaskArgumentAccessor const &);

  bool operator==(GenericTaskImplFunction const &) const;
  bool operator!=(GenericTaskImplFunction const &) const;
  bool operator<(GenericTaskImplFunction const &) const;
  bool operator>(GenericTaskImplFunction const &) const;
  bool operator<=(GenericTaskImplFunction const &) const;
  bool operator>=(GenericTaskImplFunction const &) const;
};

std::string format_as(GenericTaskImplFunction const &x);
std::ostream &operator<<(std::ostream &s, GenericTaskImplFunction const &x);

} // namespace FlexFlow

namespace std {
template <>
struct hash<::FlexFlow::GenericTaskImplFunction> {
  size_t operator()(::FlexFlow::GenericTaskImplFunction const &) const;
};
} // namespace std

#endif
