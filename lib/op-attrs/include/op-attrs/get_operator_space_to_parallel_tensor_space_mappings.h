#ifndef _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_GET_OPERATOR_SPACE_TO_PARALLEL_TENSOR_SPACE_MAPPINGS_H
#define _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_GET_OPERATOR_SPACE_TO_PARALLEL_TENSOR_SPACE_MAPPINGS_H

#include "op-attrs/computation_graph_op_attrs.dtg.h"
#include "op-attrs/num_ptensor_parallel_dims_t.h"
#include "op-attrs/operator_space_to_parallel_tensor_space_mapping.dtg.h"
#include <vector>
namespace FlexFlow {

std::vector<OperatorSpaceToParallelTensorSpaceMapping> 
  get_operator_to_input_mappings(ComputationGraphOpAttrs const &,
                                 std::vector<num_ptensor_parallel_dims_t> const &inputs_num_dims);

std::vector<OperatorSpaceToParallelTensorSpaceMapping>
  get_operator_to_output_mappings(ComputationGraphOpAttrs const &,
                                  std::vector<num_ptensor_parallel_dims_t> const &inputs_num_dims);

} // namespace FlexFlow

#endif
