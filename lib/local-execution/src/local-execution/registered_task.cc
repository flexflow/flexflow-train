#include "local-execution/registered_task.h"

namespace FlexFlow {

registered_task_t make_noop_registered_task() {
  return registered_task_t{std::monostate{}};
}

} // namespace FlexFlow
