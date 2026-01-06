#ifndef _FLEXFLOW_LIB_LOCAL_EXECUTION_INCLUDE_LOCAL_EXECUTION_PER_DEVICE_OP_STATE_INITIALIZATION_H
#define _FLEXFLOW_LIB_LOCAL_EXECUTION_INCLUDE_LOCAL_EXECUTION_PER_DEVICE_OP_STATE_INITIALIZATION_H

#include "task-spec/dynamic_graph/dynamic_open_dataflow_graph.dtg.h"
namespace FlexFlow {

DynamicOpenDataflowGraph
  perform_per_device_op_state_initialization(
    DynamicOpenDataflowGraph const &);

} // namespace FlexFlow

#endif
