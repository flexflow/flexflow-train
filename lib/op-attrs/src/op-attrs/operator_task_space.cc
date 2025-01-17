#include "op-attrs/operator_task_space.h"
#include "utils/containers/cartesian_product.h"
#include "utils/containers/maximum.h"
#include "utils/containers/product.h"
#include "utils/containers/range.h"
#include "utils/containers/transform.h"
#include "utils/containers/unordered_set_of.h"
#include "utils/fmt/unordered_set.h"
#include "utils/orthotope/orthotope.h"

namespace FlexFlow {

std::unordered_set<TaskSpaceCoordinate>
    get_task_space_coordinates(OperatorTaskSpace const &task) {

  return transform(orthotope_get_contained_coordinates(task.raw_orthotope), 
                   [](OrthotopeCoordinate const &c) { return TaskSpaceCoordinate{c}; });
}

TaskSpaceCoordinate
    get_task_space_maximum_coordinate(OperatorTaskSpace const &task) {
  return maximum(get_task_space_coordinates(task));
}

size_t num_dims(OperatorTaskSpace const &task) {
  return orthotope_num_dims(task.raw_orthotope);
}

size_t num_tasks(OperatorTaskSpace const &task) {
  return orthotope_get_volume(task.raw_orthotope);
}

} // namespace FlexFlow
