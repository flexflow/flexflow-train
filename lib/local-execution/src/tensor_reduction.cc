#include "local-execution/tensor_reduction.h"
#include "utils/containers/transform.h"

namespace FlexFlow {

reduced_tensor_t lower(tensor_guid_t const &tensor_guid) {
  return reduced_tensor_t{tensor_guid.raw_graph_output.node.raw_uid};
}

} // namespace FlexFlow
