#ifndef _FLEXFLOW_LIB_REALM_EXECUTION_INCLUDE_REALM_EXECUTION_INSTANCE_ALLOCATION_H
#define _FLEXFLOW_LIB_REALM_EXECUTION_INCLUDE_REALM_EXECUTION_INSTANCE_ALLOCATION_H

#include "realm-execution/realm_context.h"
#include "realm-execution/tensor_instance_backing.dtg.h"
#include "task-spec/dynamic_graph/dynamic_open_dataflow_graph.dtg.h"

namespace FlexFlow {

/**
 * @brief Allocates a (potentially remote) Realm instance for \p value
 * on the device represented by \p device_coord.
 */
std::pair<Realm::RegionInstance, Realm::Event>
    perform_instance_allocation_for_value(
        MachineSpaceCoordinate const &device_coord,
        DynamicValueAttrs const &value,
        RealmContext &ctx);

/**
 * @brief Allocates the (potentially remote) Realm instances for all of the
 * values in \p g, excluding the preallocated values in \p preallocated,
 * using \ref perform_instance_allocation_for_value.
 *
 * \relates TensorInstanceBacking
 */
TensorInstanceBacking perform_instance_allocation(
    DynamicOpenDataflowGraph const &g,
    std::unordered_map<DynamicValueAttrs, DynamicTensorAccessor> const
        &preallocated,
    RealmContext &ctx);

/**
 * @brief Destroys all of the instances held in \p instances.
 */
void destroy_instances(TensorInstanceBacking const &instances,
                       Realm::Event precondition);

} // namespace FlexFlow

#endif
