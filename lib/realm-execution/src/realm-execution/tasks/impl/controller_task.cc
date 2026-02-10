#include "realm-execution/tasks/impl/op_task.h"
#include "realm-execution/tasks/task_id_t.h"

namespace FlexFlow {

struct ControllerTaskArgs {
public:
  std::function<void(RealmContext &)> thunk;
};

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

Realm::Event collective_spawn_controller_task(
    RealmContext &ctx,
    Realm::Processor &target_proc,
    std::function<void(RealmContext &)> thunk) {
  ControllerTaskArgs task_args;
  task_args.thunk = thunk;

  return ctx.collective_spawn_task(target_proc,
                                   task_id_t::CONTROLLER_TASK_ID,
                                   &task_args,
                                   sizeof(task_args));
}

} // namespace FlexFlow
