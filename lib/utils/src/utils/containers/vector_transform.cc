#include "utils/containers/vector_transform.h"
#include "utils/archetypes/value_type.h"

namespace FlexFlow {

using In = value_type<0>;
using Out = value_type<1>;
using F = std::function<Out(In const &)>;

template std::vector<Out> vector_transform(std::vector<In> const &, F const &);

} // namespace FlexFlow
