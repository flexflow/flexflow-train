#include "op-attrs/task_space_coordinate.h"

namespace FlexFlow {

TaskSpaceCoordinate
    make_task_space_coordinate(std::vector<nonnegative_int> const &elems) {
  return TaskSpaceCoordinate{OrthotopeCoord{elems}};
}

} // namespace FlexFlow
