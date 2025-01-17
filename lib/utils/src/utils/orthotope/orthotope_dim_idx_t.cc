#include "utils/orthotope/orthotope_dim_idx_t.h"
#include "utils/containers/set_of.h"
#include "utils/containers/transform.h"
#include "utils/containers/range.h"

namespace FlexFlow {

std::set<orthotope_dim_idx_t> dim_idxs_for_orthotope_with_num_dims(int num_dims) {
  return set_of(transform(range(num_dims), [](int idx) { return orthotope_dim_idx_t{idx}; }));
}

} // namespace FlexFlow
