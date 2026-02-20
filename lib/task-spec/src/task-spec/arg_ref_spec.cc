#include "task-spec/arg_ref_spec.h"
#include "utils/archetypes/value_type.h"

namespace FlexFlow {

using LABEL_TYPE = value_type<0>;

template struct ArgRefSpec<LABEL_TYPE>;

} // namespace FlexFlow
