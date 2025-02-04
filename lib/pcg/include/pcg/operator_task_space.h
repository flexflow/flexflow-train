#ifndef _FLEXFLOW_PCG_INCLUDE_OPERATOR_TASK_SPACE_H
#define _FLEXFLOW_PCG_INCLUDE_OPERATOR_TASK_SPACE_H

#include "pcg/operator_task_space.dtg.h"
#include "pcg/parallel_computation_graph/parallel_computation_graph.dtg.h"
#include "pcg/parallel_computation_graph/parallel_layer_guid_t.dtg.h"
#include "pcg/task_space_coordinate.dtg.h"
#include <cstddef>
#include <unordered_set>

namespace FlexFlow {

std::unordered_set<TaskSpaceCoordinate>
    get_task_space_coordinates(OperatorTaskSpace const &task);

TaskSpaceCoordinate
    get_task_space_maximum_coordinate(OperatorTaskSpace const &task);

nonnegative_int num_dims(OperatorTaskSpace const &task);
nonnegative_int num_tasks(OperatorTaskSpace const &task);

OperatorTaskSpace get_operator_task_space(ParallelComputationGraph const &pcg,
                                          parallel_layer_guid_t const &layer);

} // namespace FlexFlow

#endif
