#ifndef _FLEXFLOW_LIB_REALM_EXECUTION_INCLUDE_REALM_EXECUTION_TASKS_IMPL_CONTROLLER_TASK_H
#define _FLEXFLOW_LIB_REALM_EXECUTION_INCLUDE_REALM_EXECUTION_TASKS_IMPL_CONTROLLER_TASK_H

#include "realm-execution/realm.h"
#include "realm-execution/realm_context.h"
#include "realm-execution/tasks/impl/controller_task_args.dtg.h"
#include <memory>

namespace FlexFlow {

/**
 * \brief Holds the result of launching a controller task via \ref
 * collective_spawn_controller_task. Owns the heap-allocated \ref
 * ControllerTaskArgs so they remain valid until the task completes. The
 * destructor automatically waits for the controller to finish before freeing
 * the args, preventing use-after-free while the controller is running.
 *
 * \note Users must explicitly block by waiting on the result before mutating
 * the contents of any values captured (e.g., by closure) in the controller.
 * Otherwise the controller may race with the caller.
 */
struct ControllerTaskResult {
public:
  explicit ControllerTaskResult(std::unique_ptr<ControllerTaskArgs> args,
                                Realm::Event event);

  ControllerTaskResult(ControllerTaskResult const &) = delete;
  ControllerTaskResult(ControllerTaskResult &&) = delete;
  ControllerTaskResult &operator=(ControllerTaskResult const &) = delete;
  ControllerTaskResult &operator=(ControllerTaskResult &&) = delete;

  /**
   * \brief Block until the controller task completes. Must be called before
   * mutating any data captured by the controller thunk to avoid data races.
   */
  void wait();

  ~ControllerTaskResult();

private:
  std::unique_ptr<ControllerTaskArgs> args;
  Realm::Event event;
};

/**
 * \brief A stub function to work around Realm not allowing lambdas to be be
 * registered as Realm tasks. Takes the desired lambda to run as the \ref
 * term-controller as an argument and immediately calls it.
 */
void controller_task_body(
    void const *, size_t, void const *, size_t, Realm::Processor);

/**
 * \brief Dispatches the \ref term-controller task. Packages up the provided \c
 * std::function and passes it along to \ref controller_task_body.
 */
ControllerTaskResult
    collective_spawn_controller_task(RealmContext &ctx,
                                     Realm::Processor &target_proc,
                                     std::function<void(RealmContext &)> thunk,
                                     Realm::Event precondition);

} // namespace FlexFlow

#endif
