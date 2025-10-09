#include "task-spec/device_specific.h"
#include "utils/archetypes/value_type.h"

namespace FlexFlow {

using T = value_type<0>;

template struct DeviceSpecific<T>;

} // namespace FlexFlow
