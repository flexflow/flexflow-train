#include "realm-execution/tasks/realm_tasks.h"
#include "realm-execution/realm_context.h"

namespace FlexFlow {

void op_task_body(
    void const *, size_t, void const *, size_t, Realm::Processor) {
  NOT_IMPLEMENTED();
}

void controller_task_body(void const *args,
                          size_t arglen,
                          void const *userdata,
                          size_t userlen,
                          Realm::Processor proc) {
  ASSERT(arglen == sizeof(std::function<void(RealmContext &)>));
  std::function<void(RealmContext &)> thunk =
      *reinterpret_cast<std::function<void(RealmContext &)> const *>(args);

  RealmContext ctx{proc};
  thunk(ctx);
}

} // namespace FlexFlow
