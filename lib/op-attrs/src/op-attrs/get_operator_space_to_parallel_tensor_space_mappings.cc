#include "op-attrs/get_operator_space_to_parallel_tensor_space_mappings.h"
#include "op-attrs/ops/element_binary.h"
#include "utils/containers/get_only.h"
#include "utils/overload.h"
#include "op-attrs/ops/linear.h"
#include "op-attrs/ops/element_unary.h"

namespace FlexFlow {

std::vector<OperatorSpaceToParallelTensorSpaceMapping> 
  get_operator_to_input_mappings(
    ComputationGraphOpAttrs const &comp_graph_op_attrs,
    std::vector<num_ptensor_parallel_dims_t> const &inputs_num_dims) {
  return comp_graph_op_attrs.visit<
    std::vector<OperatorSpaceToParallelTensorSpaceMapping>
  >(overload {
      [&](ElementBinaryAttrs const &attrs) {
        ASSERT(inputs_num_dims.size() == 2); 

        num_ptensor_parallel_dims_t lhs_num_dims = inputs_num_dims.at(0);
        num_ptensor_parallel_dims_t rhs_num_dims = inputs_num_dims.at(1);

        return std::vector{
          get_operator_to_lhs_input_mapping(attrs, lhs_num_dims, rhs_num_dims),
          get_operator_to_rhs_input_mapping(attrs, lhs_num_dims, rhs_num_dims),
        };
      },
      [&](ElementUnaryAttrs const &attrs) {
        return std::vector{
          get_operator_to_input_mapping(attrs, get_only(inputs_num_dims)),
        };
      },
      [](InputAttrs const &) { 
        return std::vector<OperatorSpaceToParallelTensorSpaceMapping>{};
      },
      [&](LinearAttrs const &attrs) {
        num_ptensor_parallel_dims_t input_num_dims = get_only(inputs_num_dims);

        std::vector<OperatorSpaceToParallelTensorSpaceMapping> result = {
          get_operator_to_input_mapping(attrs, input_num_dims),
          get_operator_to_projection_mapping(attrs, input_num_dims),
        };

        if (attrs.use_bias) {
          result.push_back(get_operator_to_bias_mapping(attrs, input_num_dims));
        };

        return result;
      },
      [](WeightAttrs const &) { 
        return std::vector<OperatorSpaceToParallelTensorSpaceMapping>{};
      },
      [](auto const &attrs) -> std::vector<OperatorSpaceToParallelTensorSpaceMapping> {
        PANIC("Missing implmentation of get_operator_to_input_mappings", attrs);
      },
  });
}

std::vector<OperatorSpaceToParallelTensorSpaceMapping>
  get_operator_to_output_mappings(
    ComputationGraphOpAttrs const &comp_graph_op_attrs,
    std::vector<num_ptensor_parallel_dims_t> const &inputs_num_dims) {

  return comp_graph_op_attrs.visit<
    std::vector<OperatorSpaceToParallelTensorSpaceMapping>
  >(overload {
      [&](ElementBinaryAttrs const &attrs) {
        ASSERT(inputs_num_dims.size() == 2); 

        num_ptensor_parallel_dims_t lhs_num_dims = inputs_num_dims.at(0);
        num_ptensor_parallel_dims_t rhs_num_dims = inputs_num_dims.at(1);

        return std::vector{
          get_operator_to_output_mapping(attrs, lhs_num_dims, rhs_num_dims),
        };
      },
      [&](ElementUnaryAttrs const &attrs) {
        return std::vector{
          get_operator_to_output_mapping(attrs, get_only(inputs_num_dims)),
        };
      },
      [&](LinearAttrs const &attrs) {
        return std::vector{
          get_operator_to_output_mapping(attrs, get_only(inputs_num_dims)),
        };
      },
      [](auto const &attrs) -> std::vector<OperatorSpaceToParallelTensorSpaceMapping> {
        PANIC("Missing implmentation of get_operator_to_input_mappings", attrs);
      },
  });
}



} // namespace FlexFlow
