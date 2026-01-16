#include "utils/archetypes/rapidcheckable_value_type.h"

namespace FlexFlow {

template struct rapidcheckable_value_type<0>;

} // namespace FlexFlow

namespace rc {

template struct Arbitrary<::FlexFlow::rapidcheckable_value_type<0>>;

}

namespace std {

template struct hash<::FlexFlow::rapidcheckable_value_type<0>>;

}
