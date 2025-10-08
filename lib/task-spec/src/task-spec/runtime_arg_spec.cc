#include "task-spec/runtime_arg_spec.h"
#include "utils/overload.h"

namespace FlexFlow {

std::type_index get_type_index(RuntimeArgSpec const &task_arg_spec) {
  return task_arg_spec.visit<std::type_index>(
      overload{[](auto const &e) { return e.get_type_index(); }});
}

} // namespace FlexFlow
