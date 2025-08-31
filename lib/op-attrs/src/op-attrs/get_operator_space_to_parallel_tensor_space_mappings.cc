#include "op-attrs/get_operator_space_to_parallel_tensor_space_mappings.h"
#include "op-attrs/ops/element_binary.h"
#include "utils/containers/get_only.h"
#include "utils/overload.h"
#include "op-attrs/ops/linear.h"
#include "op-attrs/ops/element_unary.h"

namespace FlexFlow {

std::vector<OperatorSpaceToParallelTensorSpaceMapping> 
  get_operator_to_incoming_mappings(
    ComputationGraphOpAttrs const &comp_graph_op_attrs,
    std::vector<ParallelTensorDimDegrees> const &inputs_degrees) {
  return comp_graph_op_attrs.visit<
    std::vector<OperatorSpaceToParallelTensorSpaceMapping>
  >(overload {
      [&](ElementBinaryAttrs const &attrs) {
        ASSERT(inputs_degrees.size() == 2); 

        ParallelTensorDimDegrees lhs_degrees = inputs_degrees.at(0);
        ParallelTensorDimDegrees rhs_degrees = inputs_degrees.at(1);

        return std::vector{
          get_operator_to_lhs_input_mapping(attrs, lhs_degrees, rhs_degrees),
          get_operator_to_rhs_input_mapping(attrs, lhs_degrees, rhs_degrees),
        };
      },
      [&](ElementUnaryAttrs const &attrs) {
        return std::vector{
          get_operator_to_input_mapping(attrs, get_only(inputs_degrees)),
        };
      },
      [](InputAttrs const &) { 
        return std::vector<OperatorSpaceToParallelTensorSpaceMapping>{};
      },
      [&](LinearAttrs const &attrs) {
        ParallelTensorDimDegrees input_degrees = get_only(inputs_degrees);

        std::vector<OperatorSpaceToParallelTensorSpaceMapping> result = {
          get_operator_to_input_mapping(attrs, input_degrees),
          get_operator_to_projection_mapping(attrs, input_degrees),
        };

        if (attrs.use_bias) {
          result.push_back(get_operator_to_bias_mapping(attrs, input_degrees));
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
    std::vector<ParallelTensorDimDegrees> const &inputs_degrees) {

  return comp_graph_op_attrs.visit<
    std::vector<OperatorSpaceToParallelTensorSpaceMapping>
  >(overload {
      [&](ElementBinaryAttrs const &attrs) {
        ASSERT(inputs_degrees.size() == 2); 

        ParallelTensorDimDegrees lhs_degrees = inputs_degrees.at(0);
        ParallelTensorDimDegrees rhs_degrees = inputs_degrees.at(1);

        return std::vector{
          get_operator_to_output_mapping(attrs, lhs_degrees, rhs_degrees),
        };
      },
      [&](ElementUnaryAttrs const &attrs) {
        return std::vector{
          get_operator_to_output_mapping(attrs, get_only(inputs_degrees)),
        };
      },
      [&](LinearAttrs const &attrs) {
        return std::vector{
          get_operator_to_output_mapping(attrs, get_only(inputs_degrees)),
        };
      },
      [](auto const &attrs) -> std::vector<OperatorSpaceToParallelTensorSpaceMapping> {
        PANIC("Missing implmentation of get_operator_to_input_mappings", attrs);
      },
  });
}



} // namespace FlexFlow
