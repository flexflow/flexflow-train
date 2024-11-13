#include "local-execution/tensor_reduction.h"
#include "utils/containers/transform.h"

namespace FlexFlow {

reduced_tensor_t lower(tensor_guid_t const &tensor_guid) {
  return reduced_tensor_t{tensor_guid.raw_graph_output.idx};
}

std::vector<reduced_tensor_t>
    lower(std::vector<tensor_guid_t> const &tensor_guids) {
  return transform(tensor_guids, [&](tensor_guid_t const &tensor_guid) {
    return lower(tensor_guid);
  });
}

} // namespace FlexFlow
