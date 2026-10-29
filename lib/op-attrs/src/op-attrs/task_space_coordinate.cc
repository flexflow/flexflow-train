#include "op-attrs/task_space_coordinate.h"
#include "op-attrs/operator_task_space.h"
#include "op-attrs/operator_task_space_dim_idx_t.h"
#include "op-attrs/parallel_tensor_dim_degrees.h"
#include "op-attrs/parallel_tensor_dim_idx_t.dtg.h"
#include "op-attrs/parallel_tensor_dim_idx_t.h"
#include "op-attrs/parallel_tensor_space_coordinate.h"
#include "utils/containers/map_keys.h"
#include "utils/containers/transform.h"
#include "utils/containers/vector_from_idx_map.h"
#include "utils/nonnegative_int/nonnegative_range.h"
#include "utils/nonnegative_int/num_elements.h"
#include "utils/orthotope/dim_coord.h"
#include "utils/orthotope/orthotope_coord.h"

namespace FlexFlow {

TaskSpaceCoordinate trivial_task_space_coordinate() {
  return TaskSpaceCoordinate{
      OrthotopeCoord{
          std::vector<nonnegative_int>{},
      },
  };
}

nonnegative_int task_space_coord_num_dims(TaskSpaceCoordinate const &coord) {
  return orthotope_coord_num_dims(coord.orthotope_coord);
}

TaskSpaceCoordinate
    make_task_space_coordinate(std::vector<nonnegative_int> const &elems) {
  return TaskSpaceCoordinate{OrthotopeCoord{elems}};
}

TaskSpaceCoordinate task_space_coordinate_from_dim_coord(
    DimCoord<operator_task_space_dim_idx_t> const &dim_coord) {
  std::set<operator_task_space_dim_idx_t> coord_dims =
      get_coord_dims(dim_coord);

  std::set<operator_task_space_dim_idx_t> dims =
      operator_task_space_dim_idx_range(num_elements(coord_dims));

  ASSERT(coord_dims == dims);

  std::map<nonnegative_int, nonnegative_int> idx_map =
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
      operator_task_space_dim_idx_range(
          orthotope_coord_num_dims(coord.orthotope_coord)),
      get_operator_task_space_dim_ordering());
}

TaskSpaceCoordinate task_coord_matching_parallel_tensor_space_coordinate(
    ParallelTensorSpaceCoordinate const &output_coord,
    ParallelTensorDimDegrees const &output_degrees) {
  ASSERT(parallel_tensor_space_contains_coord(output_degrees, output_coord));

  std::set<parallel_tensor_dim_idx_t> nontrivial_output_dims =
      get_nontrivial_parallel_tensor_dim_indices(output_degrees);

  return TaskSpaceCoordinate{
      orthotope_coord_from_dim_coord(
          restrict_coord_to_dims(
              dim_coord_from_parallel_tensor_space_coord(output_coord),
              nontrivial_output_dims),
          get_parallel_tensor_dim_ordering()),
  };
}

bool task_space_coord_set_is_orthotopic(
    std::set<TaskSpaceCoordinate> const &coord_set) {
  std::set<DimCoord<operator_task_space_dim_idx_t>> dim_coord_set =
      transform(coord_set,
                [&](TaskSpaceCoordinate const &c)
                    -> DimCoord<operator_task_space_dim_idx_t> {
                  return dim_coord_from_task_space_coordinate(c);
                });

  std::optional<DimDomain<operator_task_space_dim_idx_t>> dim_domain =
      strict_dim_domain_for_coord_set(dim_coord_set);

  return dim_domain.has_value();
}

} // namespace FlexFlow
