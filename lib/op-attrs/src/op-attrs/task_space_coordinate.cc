#include "op-attrs/task_space_coordinate.h"
#include "op-attrs/operator_task_space.h"
#include "op-attrs/operator_task_space_dim_idx_t.h"
#include "utils/containers/map_keys.h"
#include "utils/containers/transform.h"
#include "utils/containers/unordered_set_of.h"
#include "utils/containers/vector_from_idx_map.h"
#include "utils/nonnegative_int/nonnegative_range.h"
#include "utils/nonnegative_int/num_elements.h"
#include "utils/orthotope/dim_coord.h"
#include "utils/orthotope/orthotope_coord.h"

namespace FlexFlow {

nonnegative_int task_space_coord_num_dims(TaskSpaceCoordinate const &coord) {
  return orthotope_coord_num_dims(coord.orthotope_coord);
}

TaskSpaceCoordinate
    make_task_space_coordinate(std::vector<nonnegative_int> const &elems) {
  return TaskSpaceCoordinate{OrthotopeCoord{elems}};
}

TaskSpaceCoordinate task_space_coordinate_from_dim_coord(
    DimCoord<operator_task_space_dim_idx_t> const &dim_coord) {
  std::unordered_set<operator_task_space_dim_idx_t> coord_dims =
      get_coord_dims(dim_coord);

  std::set<operator_task_space_dim_idx_t> dims =
      operator_task_space_dim_idx_range(num_elements(coord_dims));

  ASSERT(coord_dims == unordered_set_of(dims));

  std::unordered_map<nonnegative_int, nonnegative_int> idx_map =
      map_keys(dim_coord.raw,
               [](operator_task_space_dim_idx_t idx) { return idx.raw_idx; });

  return TaskSpaceCoordinate{
      OrthotopeCoord{
          vector_from_idx_map(idx_map).value(),
      },
  };
}

DimCoord<operator_task_space_dim_idx_t>
    dim_coord_from_task_space_coordinate(TaskSpaceCoordinate const &coord) {

  return dim_coord_from_orthotope_coord(
      coord.orthotope_coord,
      unordered_set_of(operator_task_space_dim_idx_range(
          orthotope_coord_num_dims(coord.orthotope_coord))),
      get_operator_task_space_dim_ordering());
}

} // namespace FlexFlow
