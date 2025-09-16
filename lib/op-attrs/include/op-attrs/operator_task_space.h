#ifndef _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_OPERATOR_TASK_SPACE_H
#define _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_OPERATOR_TASK_SPACE_H

#include "op-attrs/operator_task_space.dtg.h"
#include "op-attrs/operator_task_space_dim_idx_t.dtg.h"
#include "op-attrs/parallel_tensor_dim_degrees.dtg.h"
#include "op-attrs/task_space_coordinate.dtg.h"
#include "utils/orthotope/dim_domain.dtg.h"
#include "utils/orthotope/dim_ordering.dtg.h"
#include "utils/orthotope/minimal_dim_domain.dtg.h"
#include <unordered_set>

namespace FlexFlow {

OperatorTaskSpace trivial_op_task_space();

std::unordered_set<operator_task_space_dim_idx_t> 
    operator_task_space_get_dim_idxs(OperatorTaskSpace const &);

std::unordered_set<TaskSpaceCoordinate>
    get_task_space_coordinates(OperatorTaskSpace const &operator_task_space);

bool operator_task_space_contains_coord(OperatorTaskSpace const &,
                                        TaskSpaceCoordinate const &);

TaskSpaceCoordinate
    get_task_space_maximum_coordinate(OperatorTaskSpace const &operator_task_space);

nonnegative_int op_task_space_num_dims(OperatorTaskSpace const &operator_task_space);
positive_int num_tasks(OperatorTaskSpace const &operator_task_space);

positive_int op_task_space_dim_size_for_idx(OperatorTaskSpace const &,
                                            operator_task_space_dim_idx_t);

MinimalDimDomain<operator_task_space_dim_idx_t>
  minimal_dim_domain_from_operator_task_space(OperatorTaskSpace const &);

OperatorTaskSpace
  operator_task_space_from_minimal_dim_domain(MinimalDimDomain<operator_task_space_dim_idx_t> const &); 

DimOrdering<operator_task_space_dim_idx_t>
  get_operator_task_space_dim_ordering();

OperatorTaskSpace 
  get_operator_task_space_matching_parallel_tensor_dim_degrees(
    ParallelTensorDimDegrees const &dim_degrees); 

} // namespace FlexFlow

#endif
