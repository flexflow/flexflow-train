#include "task-spec/arg_ref.h"
#include "utils/archetypes/value_type.h"

namespace FlexFlow {

using LABEL_TYPE = value_type<0>;
using T = value_type<1>;

template struct ArgRef<LABEL_TYPE, T>;

} // namespace FlexFlow
