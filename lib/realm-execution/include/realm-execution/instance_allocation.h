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
 * \brief Perform instance allocation with pre-created Realm instances.
 *
 * Used for ExternalTensorBinding — the Realm instance already exists
 * (created by create_external_tensor) and should be inserted directly
 * into the backing without re-creating it.
 *
 * \param preallocated_instances Map of DynamicValueAttrs to already-created
 *        (RegionInstance, Event) pairs. Takes precedence over preallocated.
 */
TensorInstanceBacking perform_instance_allocation(
    DynamicOpenDataflowGraph const &g,
    std::unordered_map<DynamicValueAttrs, DynamicTensorAccessor> const
        &preallocated,
    std::unordered_map<DynamicValueAttrs,
                       std::pair<Realm::RegionInstance, Realm::Event>> const
        &preallocated_instances,
    RealmContext &ctx);

/**
 * @brief Destroys all of the instances held in \p instances.
 */
void destroy_instances(TensorInstanceBacking const &instances,
                       Realm::Event precondition);

} // namespace FlexFlow

#endif
