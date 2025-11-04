#include "utils/required.h"
#include "utils/archetypes/jsonable_value_type.h"

using T = ::FlexFlow::jsonable_value_type<0>;

namespace nlohmann {

template struct adl_serializer<::FlexFlow::req<T>>;

} // namespace FlexFlow
