#include "local-execution/task_id_with_noop_default_t.h"

namespace FlexFlow {

task_id_with_noop_default_t make_noop_registered_task() {
  return task_id_with_noop_default_t{std::monostate{}};
}

} // namespace FlexFlow
