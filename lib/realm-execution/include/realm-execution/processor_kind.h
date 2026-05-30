#ifndef _FLEXFLOW_LIB_REALM_EXECUTION_INCLUDE_REALM_EXECUTION_PROCESSOR_KIND_H
#define _FLEXFLOW_LIB_REALM_EXECUTION_INCLUDE_REALM_EXECUTION_PROCESSOR_KIND_H

#include "pcg/device_type.dtg.h"
#include "realm-execution/realm.h"

namespace FlexFlow {

DeviceType device_type_from_processor_kind(Realm::Processor::Kind);
Realm::Processor::Kind processor_kind_from_device_type(DeviceType);

} // namespace FlexFlow

#endif
