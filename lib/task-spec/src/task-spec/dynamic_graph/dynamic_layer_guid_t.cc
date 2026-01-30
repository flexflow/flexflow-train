#include "task-spec/dynamic_graph/dynamic_layer_guid_t.h"
#include "task-spec/dynamic_graph/dynamic_loss_layer_guid_t.dtg.h"

namespace FlexFlow {

dynamic_layer_guid_t mk_dynamic_layer_guid(layer_guid_t l) {
  return dynamic_layer_guid_t{l};
}
dynamic_layer_guid_t mk_dynamic_layer_guid_parallel(parallel_layer_guid_t l) {
  return dynamic_layer_guid_t{l};
}
dynamic_layer_guid_t mk_dynamic_layer_guid_loss() {
  return dynamic_layer_guid_t{dynamic_loss_layer_guid_t{}};
}

} // namespace FlexFlow
