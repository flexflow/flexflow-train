#include "realm-execution/device_specific_ptr.h"
#include "utils/archetypes/value_type.h"

namespace FlexFlow {

using T = value_type<0>;

template struct DeviceSpecificPtr<T>;

} // namespace FlexFlow
