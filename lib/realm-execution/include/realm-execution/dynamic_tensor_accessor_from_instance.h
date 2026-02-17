#ifndef _FLEXFLOW_LIB_REALM_EXECUTION_INCLUDE_REALM_EXECUTION_DYNAMIC_TENSOR_ACCESSOR_FROM_INSTANCE_H
#define _FLEXFLOW_LIB_REALM_EXECUTION_INCLUDE_REALM_EXECUTION_DYNAMIC_TENSOR_ACCESSOR_FROM_INSTANCE_H

#include "realm-execution/realm.h"
#include "task-spec/dynamic_graph/dynamic_tensor_accessor.dtg.h"

namespace FlexFlow {

DynamicTensorAccessor
    dynamic_tensor_accessor_from_instance(Realm::RegionInstance const &);

} // namespace FlexFlow

#endif
