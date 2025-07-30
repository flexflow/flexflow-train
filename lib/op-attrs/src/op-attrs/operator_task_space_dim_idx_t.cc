#include "op-attrs/operator_task_space_dim_idx_t.h"
#include "utils/containers/set_of.h"
#include "utils/containers/transform.h"
#include "utils/nonnegative_int/range.h"

namespace FlexFlow {

std::set<operator_task_space_dim_idx_t>
    operator_task_space_dim_idx_range(nonnegative_int end) {
  return transform(set_of(range(end)), [](nonnegative_int raw_idx) {
    return operator_task_space_dim_idx_t{raw_idx};
  });
}

} // namespace FlexFlow
