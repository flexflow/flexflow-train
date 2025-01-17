#include "local-execution/tensor_lowering.h"
#include "utils/containers/transform.h"

namespace FlexFlow {

lowered_tensor_t lower(tensor_guid_t const &tensor_guid) {
  return lowered_tensor_t{tensor_guid.raw_graph_output.node.raw_uid};
}

} // namespace FlexFlow
