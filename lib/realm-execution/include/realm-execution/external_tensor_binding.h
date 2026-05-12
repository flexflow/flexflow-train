#ifndef _FLEXFLOW_LIB_REALM_EXECUTION_INCLUDE_REALM_EXECUTION_EXTERNAL_TENSOR_BINDING_H
#define _FLEXFLOW_LIB_REALM_EXECUTION_INCLUDE_REALM_EXECUTION_EXTERNAL_TENSOR_BINDING_H

#include "op-attrs/parallel_tensor_space_coordinate.dtg.h"
#include "pcg/machine_space_coordinate.dtg.h"
#include "pcg/parallel_computation_graph/parallel_tensor_guid_t.dtg.h"
#include "realm-execution/external_tensor_handle.h"

namespace FlexFlow {

/**
 * \brief Binds an \ref ExternalTensorHandle to a specific tensor and
 * shard coordinate in a \ref MappedParallelComputationGraph.
 *
 * Used to pass pre-allocated input tensors to \ref create_pcg_instance
 * without exposing Realm details to the caller.
 */
struct ExternalTensorBinding {
  parallel_tensor_guid_t tensor_guid;
  ParallelTensorSpaceCoordinate shard_coord;
  MachineSpaceCoordinate machine_coord;
  ExternalTensorHandle handle;
};

} // namespace FlexFlow

#endif
