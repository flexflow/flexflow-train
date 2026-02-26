#include "task-spec/dynamic_graph/dynamic_tensor_guid_t.h"
#include "task-spec/dynamic_graph/dynamic_loss_tensor_guid_t.dtg.h"

namespace FlexFlow {

dynamic_tensor_guid_t mk_dynamic_tensor_guid_for_tensor_guid(tensor_guid_t t) {
  return dynamic_tensor_guid_t{t};
}
dynamic_tensor_guid_t
    mk_dynamic_tensor_guid_for_parallel_tensor_guid(parallel_tensor_guid_t t) {
  return dynamic_tensor_guid_t{t};
}
dynamic_tensor_guid_t mk_dynamic_tensor_guid_for_loss() {
  return dynamic_tensor_guid_t{dynamic_loss_tensor_guid_t{}};
}

} // namespace FlexFlow
