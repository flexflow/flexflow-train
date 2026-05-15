#include "realm-execution/tasks/serializer/task_arg_serializer.h"
#include "utils/archetypes/jsonable_value_type.h"

namespace FlexFlow {

using T = jsonable_value_type<0>;

template std::string serialize_task_args(T const &);

template T deserialize_task_args(void const *, size_t);

} // namespace FlexFlow
