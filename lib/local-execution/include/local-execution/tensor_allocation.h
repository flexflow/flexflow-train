#ifndef _FLEXFLOW_LIB_LOCAL_EXECUTION_INCLUDE_LOCAL_EXECUTION_TENSOR_ALLOCATION_H
#define _FLEXFLOW_LIB_LOCAL_EXECUTION_INCLUDE_LOCAL_EXECUTION_TENSOR_ALLOCATION_H

#include "kernels/allocation.h"
#include "task-spec/dynamic_graph/dynamic_open_dataflow_graph.dtg.h"

namespace FlexFlow {

bool no_tensors_are_allocated(DynamicOpenDataflowGraph const &);
bool all_tensors_are_allocated(DynamicOpenDataflowGraph const &);

DynamicValueAttrs
  perform_tensor_allocation_for_value(DynamicValueAttrs const &, 
                                      Allocator &);

DynamicOpenDataflowGraph
  perform_tensor_allocation(DynamicOpenDataflowGraph const &, 
                            std::unordered_map<DynamicValueAttrs, GenericTensorAccessorW> const &preallocated,
                            Allocator &);

} // namespace FlexFlow

#endif
