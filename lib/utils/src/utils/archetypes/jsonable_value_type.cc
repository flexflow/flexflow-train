#include "utils/archetypes/jsonable_value_type.h"

namespace FlexFlow {

template struct jsonable_value_type<0>;

} // namespace FlexFlow

namespace nlohmann {

template struct adl_serializer<::FlexFlow::jsonable_value_type<0>>;

} // namespace nlohmann

namespace std {

template struct hash<::FlexFlow::jsonable_value_type<0>>;

}
