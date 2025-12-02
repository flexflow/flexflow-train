#include "task-spec/ops/impl/broadcast.h"

namespace FlexFlow {

OpTaskInvocation forward(BroadcastAttrs const &) {
  NOT_IMPLEMENTED();
}

OpTaskInvocation backward(BroadcastAttrs const &) {
  NOT_IMPLEMENTED();
}

TaskImplFunction get_broadcast_fwd_task_impl() {
  NOT_IMPLEMENTED();
}

TaskImplFunction get_broadcast_bwd_task_impl() {
  NOT_IMPLEMENTED();
}

} // namespace FlexFlow
