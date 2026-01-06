#include "task-spec/device_specific.h"
#include "utils/archetypes/value_type.h"

using T = ::FlexFlow::value_type<0>;

namespace FlexFlow {

template struct DeviceSpecific<T>;

} // namespace FlexFlow

namespace std {

template struct hash<::FlexFlow::DeviceSpecific<T>>;

}
