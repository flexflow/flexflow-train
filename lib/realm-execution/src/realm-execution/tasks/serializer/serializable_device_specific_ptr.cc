#include "realm-execution/tasks/serializer/serializable_device_specific_ptr.h"
#include "utils/archetypes/value_type.h"

namespace FlexFlow {

using T = value_type<0>;

template SerializableDeviceSpecificPtr
    device_specific_ptr_to_serializable(DeviceSpecificPtr<T> const &);

template DeviceSpecificPtr<T> device_specific_ptr_from_serializable(
    SerializableDeviceSpecificPtr const &);

} // namespace FlexFlow
