#include "realm-execution/tasks/impl/controller_task.h"
#include "realm-execution/tasks/impl/op_task.h"
#include "realm-execution/tasks/task_id_t.h"
#include <cstring>

namespace FlexFlow {

ControllerTaskResult::ControllerTaskResult(
    std::unique_ptr<ControllerTaskArgs> args, Realm::Event event)
    : args(std::move(args)), event(event) {}

void ControllerTaskResult::wait() {
  this->event.wait();
}

ControllerTaskResult::~ControllerTaskResult() {
  if (this->args != nullptr) {
    this->event.wait();
  }
}

void controller_task_body(void const *args,
                          size_t arglen,
                          void const *userdata,
                          size_t userlen,
                          Realm::Processor proc) {
  ASSERT(arglen == sizeof(ControllerTaskArgs *));
  ControllerTaskArgs *task_args_ptr;
  std::memcpy(&task_args_ptr, args, sizeof(ControllerTaskArgs *));

  RealmContext ctx{proc};
  task_args_ptr->thunk(ctx);
}

ControllerTaskResult
    collective_spawn_controller_task(RealmContext &ctx,
                                     Realm::Processor &target_proc,
                                     std::function<void(RealmContext &)> thunk,
                                     Realm::Event precondition) {
  // ControllerTaskArgs are passed by pointer because they are NOT trivially
  // copyable, and we use the ControllerTaskResult to manage the lifetime of the
  // data to avoid use-after-free while the controller runs
  ControllerTaskArgs *raw_ptr = new ControllerTaskArgs{thunk};

  Realm::Event event = ctx.collective_spawn_task(target_proc,
                                                 task_id_t::CONTROLLER_TASK_ID,
                                                 &raw_ptr,
                                                 sizeof(raw_ptr),
                                                 precondition);

  return ControllerTaskResult{std::unique_ptr<ControllerTaskArgs>(raw_ptr),
                              event};
}

} // namespace FlexFlow
