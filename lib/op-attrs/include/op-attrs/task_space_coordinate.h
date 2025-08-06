#ifndef _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_TASK_SPACE_COORDINATE_H
#define _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_TASK_SPACE_COORDINATE_H

#include "op-attrs/operator_task_space_dim_idx_t.dtg.h"
#include "op-attrs/task_space_coordinate.dtg.h"
#include "utils/orthotope/dim_coord.dtg.h"

namespace FlexFlow {

TaskSpaceCoordinate
    make_task_space_coordinate(std::vector<nonnegative_int> const &);

TaskSpaceCoordinate 
  task_space_coordinate_from_dim_coord(
    DimCoord<operator_task_space_dim_idx_t> const &);

DimCoord<operator_task_space_dim_idx_t>
  dim_coord_from_task_space_coordinate(TaskSpaceCoordinate const &);

} // namespace FlexFlow

#endif
