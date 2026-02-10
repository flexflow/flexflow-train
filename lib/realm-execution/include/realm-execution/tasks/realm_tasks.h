#ifndef _FLEXFLOW_LIB_REALM_EXECUTION_INCLUDE_REALM_EXECUTION_REALM_TASKS_H
#define _FLEXFLOW_LIB_REALM_EXECUTION_INCLUDE_REALM_EXECUTION_REALM_TASKS_H

#include "realm-execution/realm.h"
#include "realm-execution/realm_context.h"
#include "task-spec/dynamic_graph/dynamic_node_invocation.dtg.h"
#include <type_traits>

namespace FlexFlow {

void op_task_body(void const *, size_t, void const *, size_t, Realm::Processor);

// TODO: at some point we're going to have to actually serialize these, but for
// now just pass the pointer and assume we're running inside a single address
// space
struct DeviceInitTaskArgs {
public:
  DynamicNodeInvocation *invocation;
};
static_assert(std::has_unique_object_representations_v<DeviceInitTaskArgs>);

void device_init_task_body(
    void const *, size_t, void const *, size_t, Realm::Processor);

struct ControllerTaskArgs {
public:
  std::function<void(RealmContext &)> thunk;
};

void controller_task_body(
    void const *, size_t, void const *, size_t, Realm::Processor);

} // namespace FlexFlow

#endif
