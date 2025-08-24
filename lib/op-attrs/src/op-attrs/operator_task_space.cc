#include "op-attrs/operator_task_space.h"
#include "op-attrs/operator_task_space.dtg.h"
#include "op-attrs/operator_task_space_dim_idx_t.h"
#include "op-attrs/parallel_tensor_dim_degrees.h"
#include "op-attrs/parallel_tensor_dim_idx_t.h"
#include "op-attrs/parallel_tensor_shape.dtg.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "utils/containers/cartesian_product.h"
#include "utils/containers/extend.h"
#include "utils/containers/maximum.h"
#include "utils/containers/product.h"
#include "utils/containers/range.h"
#include "utils/containers/transform.h"
#include "utils/containers/unordered_set_of.h"
#include "utils/containers/vector_of.h"
#include "utils/fmt/unordered_set.h"
#include "utils/nonnegative_int/nonnegative_range.h"
#include "utils/nonnegative_int/num_elements.h"
#include "utils/orthotope/dim_domain.h"
#include "utils/orthotope/dim_ordering.h"
#include "utils/orthotope/minimal_dim_domain.h"
#include "utils/orthotope/orthotope.dtg.h"
#include "utils/orthotope/orthotope.h"

namespace FlexFlow {

std::unordered_set<TaskSpaceCoordinate>
    get_task_space_coordinates(OperatorTaskSpace const &task) {

  std::vector<std::vector<nonnegative_int>> coordinate_ranges =
      transform(task.degrees, [&](positive_int num_points) {
        return nonnegative_range(
            num_points.nonnegative_int_from_positive_int());
      });

  std::unordered_set<std::vector<nonnegative_int>> raw_coordinates =
      unordered_set_of(cartesian_product(coordinate_ranges));
  std::unordered_set<TaskSpaceCoordinate> task_space_coordinates =
      transform(raw_coordinates, [](std::vector<nonnegative_int> const &point) {
        return TaskSpaceCoordinate{OrthotopeCoord{point}};
      });
  return task_space_coordinates;
}

TaskSpaceCoordinate
    get_task_space_maximum_coordinate(OperatorTaskSpace const &task) {
  return maximum(get_task_space_coordinates(task));
}

nonnegative_int num_dims(OperatorTaskSpace const &task) {
  return num_elements(task.degrees);
}

positive_int num_tasks(OperatorTaskSpace const &task) {
  return product(task.degrees);
}

DimDomain<operator_task_space_dim_idx_t>
  dim_domain_from_operator_task_space(OperatorTaskSpace const &operator_task_space) {
  
  Orthotope orthotope = 
    Orthotope{operator_task_space.degrees};

  return dim_domain_from_orthotope(
    orthotope,
    unordered_set_of(operator_task_space_dim_idx_range(orthotope_get_num_dims(orthotope))),
    get_operator_task_space_dim_ordering());
}

DimOrdering<operator_task_space_dim_idx_t>
  get_operator_task_space_dim_ordering() {
  return make_default_dim_ordering<operator_task_space_dim_idx_t>();
}

OperatorTaskSpace 
  get_operator_task_space_matching_parallel_tensor_dim_degrees(
    ParallelTensorDimDegrees const &dim_degrees) {
  return OperatorTaskSpace{
    minimal_orthotope_from_minimal_dim_domain(
      minimal_dim_domain_from_parallel_tensor_dim_degrees(dim_degrees),
      get_parallel_tensor_dim_ordering()),
  };
}


} // namespace FlexFlow
