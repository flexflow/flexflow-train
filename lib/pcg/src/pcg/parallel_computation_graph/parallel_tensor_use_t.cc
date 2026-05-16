#include "pcg/parallel_computation_graph/parallel_tensor_use_t.h"

namespace FlexFlow {

parallel_layer_guid_t parallel_tensor_use_get_layer(parallel_tensor_use_t const &u) {
  return parallel_layer_guid_t{u.raw_dataflow_input.node};
}

TensorSlotName parallel_tensor_use_get_slot(parallel_tensor_use_t const &u) {
  return u.raw_dataflow_input.slot_name;
}

} // namespace FlexFlow
