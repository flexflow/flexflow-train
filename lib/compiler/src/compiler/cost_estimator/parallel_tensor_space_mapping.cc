#include "compiler/cost_estimator/parallel_tensor_space_mapping.h"
#include "utils/archetypes/value_type.h"

using T = ::FlexFlow::value_type<0>;

namespace FlexFlow {

template struct ParallelTensorSpaceMapping<T>;

template
  std::string format_as(ParallelTensorSpaceMapping<T> const &);

template
  std::ostream &operator<<(std::ostream &, ParallelTensorSpaceMapping<T> const &);

} // namespace FlexFlow

namespace std {

template struct hash<::FlexFlow::ParallelTensorSpaceMapping<T>>;

}
