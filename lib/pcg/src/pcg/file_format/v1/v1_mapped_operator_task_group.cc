#include "pcg/file_format/v1/v1_mapped_operator_task_group.h"

namespace FlexFlow {

V1MappedOperatorTaskGroup to_v1(MappedOperatorTaskGroup const &g) {
  return V1MappedOperatorTaskGroup{g.get_shard_bindings()};
}

MappedOperatorTaskGroup from_v1(V1MappedOperatorTaskGroup const &v1) {
  return MappedOperatorTaskGroup{v1.shard_bindings};
}

} // namespace FlexFlow
