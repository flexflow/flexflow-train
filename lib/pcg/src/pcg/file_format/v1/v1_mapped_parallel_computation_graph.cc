#include "pcg/file_format/v1/v1_mapped_parallel_computation_graph.h"
#include "pcg/file_format/v1/v1_mapped_operator_task_group.h"
#include "pcg/file_format/v1/v1_parallel_computation_graph.h"
#include "utils/containers/map_values.h"

namespace FlexFlow {

V1MappedParallelComputationGraph
    to_v1(MappedParallelComputationGraph const &mpcg) {
  return V1MappedParallelComputationGraph{
      to_v1(mpcg.pcg),
      map_values(mpcg.mapped_tasks,
                 [](MappedOperatorTaskGroup const &g) { return to_v1(g); }),
  };
}

MappedParallelComputationGraph
    from_v1(V1MappedParallelComputationGraph const &v1) {
  return MappedParallelComputationGraph{
      from_v1(v1.pcg),
      map_values(v1.mapped_tasks,
                 [](V1MappedOperatorTaskGroup const &g) { return from_v1(g); }),
  };
}

} // namespace FlexFlow
