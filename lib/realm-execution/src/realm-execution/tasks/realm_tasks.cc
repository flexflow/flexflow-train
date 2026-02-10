#include "realm-execution/tasks/realm_tasks.h"
#include "realm-execution/realm_context.h"
#include "utils/exception.h"

namespace FlexFlow {

void op_task_body(
    void const *, size_t, void const *, size_t, Realm::Processor) {
  NOT_IMPLEMENTED();
}

void device_init_task_body(
    void const *, size_t, void const *, size_t, Realm::Processor) {
  NOT_IMPLEMENTED();
}

void controller_task_body(void const *args,
                          size_t arglen,
                          void const *userdata,
                          size_t userlen,
                          Realm::Processor proc) {
  ASSERT(arglen == sizeof(ControllerTaskArgs));
  ControllerTaskArgs task_args =
      *reinterpret_cast<ControllerTaskArgs const *>(args);

  RealmContext ctx{proc};
  task_args.thunk(ctx);
}

} // namespace FlexFlow
