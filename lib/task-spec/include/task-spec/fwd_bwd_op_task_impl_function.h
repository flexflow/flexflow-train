#ifndef _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_FWD_BWD_OP_TASK_IMPL_FUNCTION_H
#define _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_FWD_BWD_OP_TASK_IMPL_FUNCTION_H

#include "task-spec/task_argument_accessor/task_argument_accessor.h"
#include "utils/units/milliseconds_t.h"

namespace FlexFlow {

struct FwdBwdOpTaskImplFunction {

  std::optional<milliseconds_t> (*function_ptr)(TaskArgumentAccessor const &);

  bool operator==(FwdBwdOpTaskImplFunction const &) const;
  bool operator!=(FwdBwdOpTaskImplFunction const &) const;
  bool operator<(FwdBwdOpTaskImplFunction const &) const;
  bool operator>(FwdBwdOpTaskImplFunction const &) const;
  bool operator<=(FwdBwdOpTaskImplFunction const &) const;
  bool operator>=(FwdBwdOpTaskImplFunction const &) const;
};

std::string format_as(FwdBwdOpTaskImplFunction const &x);
std::ostream &operator<<(std::ostream &s, FwdBwdOpTaskImplFunction const &x);

} // namespace FlexFlow

namespace std {
template <>
struct hash<::FlexFlow::FwdBwdOpTaskImplFunction> {
  size_t operator()(::FlexFlow::FwdBwdOpTaskImplFunction const &) const;
};
} // namespace std

#endif
