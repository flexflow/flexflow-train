#ifndef _FLEXFLOW_LIB_REALM_EXECUTION_INCLUDE_REALM_EXECUTION_DYNAMIC_TENSOR_ACCESSOR_FROM_INSTANCE_H
#define _FLEXFLOW_LIB_REALM_EXECUTION_INCLUDE_REALM_EXECUTION_DYNAMIC_TENSOR_ACCESSOR_FROM_INSTANCE_H

#include "op-attrs/parallel_tensor_shape.dtg.h"
#include "realm-execution/realm.h"
#include "task-spec/dynamic_graph/dynamic_tensor_accessor.dtg.h"
#include "task-spec/permissions.h"

namespace FlexFlow {

/**
 * @brief Turn a %Realm region instance into a GenericTensorAccessor.
 */
DynamicTensorAccessor dynamic_tensor_accessor_from_instance(
    Realm::RegionInstance inst,
    Realm::Event ready,
    ParallelTensorShape const &parallel_tensor_shape,
    Permissions const &permissions,
    Realm::Processor for_processor);

} // namespace FlexFlow

#endif
