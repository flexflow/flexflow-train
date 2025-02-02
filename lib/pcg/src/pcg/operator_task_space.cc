#include "pcg/operator_task_space.h"
#include "op-attrs/parallel_tensor_shape.dtg.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "pcg/operator_task_space.dtg.h"
#include "pcg/parallel_computation_graph/parallel_computation_graph.h"
#include "pcg/parallel_computation_graph/parallel_layer_guid_t.dtg.h"
#include "pcg/parallel_computation_graph/parallel_tensor_guid_t.dtg.h"
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

namespace FlexFlow {

std::unordered_set<TaskSpaceCoordinate>
    get_task_space_coordinates(OperatorTaskSpace const &task) {

  std::vector<std::vector<nonnegative_int>> coordinate_ranges =
      transform(task.degrees, [&](nonnegative_int num_points) {
        return nonnegative_range(num_points);
      });

  std::unordered_set<std::vector<nonnegative_int>> raw_coordinates =
      unordered_set_of(cartesian_product(coordinate_ranges));
  std::unordered_set<TaskSpaceCoordinate> task_space_coordinates =
      transform(raw_coordinates, [](std::vector<nonnegative_int> const &point) {
        return TaskSpaceCoordinate{point};
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

nonnegative_int num_tasks(OperatorTaskSpace const &task) {
  return product(task.degrees);
}

OperatorTaskSpace get_operator_task_space(ParallelComputationGraph const &pcg,
                                          parallel_layer_guid_t const &layer) {
  parallel_tensor_guid_t out_tensor = get_layer_outputs(pcg, layer).at(0);
  ParallelTensorShape shape = get_parallel_tensor_shape(pcg, out_tensor);

  std::vector<nonnegative_int> degrees;
  extend(degrees, vector_of(ff_ordered_shard_degrees(shape)));
  degrees.push_back(get_sum_degree(shape));
  degrees.push_back(get_discard_copy_degree(shape));
  return OperatorTaskSpace{degrees};
}

} // namespace FlexFlow
