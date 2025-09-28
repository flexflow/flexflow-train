#include "op-attrs/get_operator_space_to_parallel_tensor_space_mappings.h"
#include "op-attrs/get_incoming_tensor_roles.h"
#include "op-attrs/ops/element_binary.h"
#include "utils/containers/filtrans.h"
#include "utils/containers/get_only.h"
#include "utils/overload.h"
#include "op-attrs/ops/linear.h"
#include "op-attrs/ops/element_unary.h"
#include "op-attrs/ops/input.h"
#include "op-attrs/ops/transpose.h"
#include "op-attrs/ops/weight.h"

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
      [&](InputAttrs const &) { 
        ASSERT(inputs_degrees.size() == 0);

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
      [&](TransposeAttrs const &attrs) {
        ASSERT(inputs_degrees.size() == 1);

        return std::vector{
          get_operator_to_input_mapping(attrs, get_only(inputs_degrees)),
        };
      },
      [&](WeightAttrs const &) { 
        ASSERT(inputs_degrees.size() == 0);

        return std::vector<OperatorSpaceToParallelTensorSpaceMapping>{};
      },
      [](auto const &attrs) -> std::vector<OperatorSpaceToParallelTensorSpaceMapping> {
        PANIC("Missing implmentation of get_operator_to_input_mappings", attrs);
      },
  });
}

std::vector<OperatorSpaceToParallelTensorSpaceMapping> 
  get_operator_to_incoming_mappings_for_role(ComputationGraphOpAttrs const &attrs,
                                             std::vector<ParallelTensorDimDegrees> const &inputs_degrees,
                                             IncomingTensorRole incoming_tensor_role) {

  std::vector<OperatorSpaceToParallelTensorSpaceMapping>
    incoming_mappings = get_operator_to_incoming_mappings(attrs, inputs_degrees);


  std::vector<IncomingTensorRole> 
    incoming_tensor_roles = get_incoming_tensor_roles(attrs, num_elements(inputs_degrees));

  return filtrans(zip(incoming_mappings, incoming_tensor_roles),
                  [&](std::pair<OperatorSpaceToParallelTensorSpaceMapping, IncomingTensorRole> const &p) 
                    -> std::optional<OperatorSpaceToParallelTensorSpaceMapping>
                  {
                    auto const &[mapping, role] = p;

                    if (role == incoming_tensor_role) {
                      return mapping;
                    } else {
                      return std::nullopt;
                    }
                  });
}

std::vector<OperatorSpaceToParallelTensorSpaceMapping>
  get_operator_to_input_mappings(ComputationGraphOpAttrs const &attrs,
                                 std::vector<ParallelTensorDimDegrees> const &inputs_degrees) {
  return get_operator_to_incoming_mappings_for_role(attrs, inputs_degrees, IncomingTensorRole::INPUT);
}

std::vector<OperatorSpaceToParallelTensorSpaceMapping>
  get_operator_to_weight_mappings(ComputationGraphOpAttrs const &attrs,
                                 std::vector<ParallelTensorDimDegrees> const &inputs_degrees) {

  return get_operator_to_incoming_mappings_for_role(attrs, inputs_degrees, IncomingTensorRole::WEIGHT);
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
      [&](InputAttrs const &attrs) {
        ASSERT(inputs_degrees.size() == 0);

        return std::vector{
          get_operator_to_output_mapping(attrs),
        };
      },
      [&](TransposeAttrs const &attrs) {
        ASSERT(inputs_degrees.size() == 1);

        return std::vector{
          get_operator_to_output_mapping(attrs, get_only(inputs_degrees)),
        };
      },
      [&](WeightAttrs const &attrs) {
        ASSERT(inputs_degrees.size() == 0);

        return std::vector{
          get_operator_to_output_mapping(attrs),
        };
      },
      [](auto const &attrs) -> std::vector<OperatorSpaceToParallelTensorSpaceMapping> {
        PANIC("Missing implmentation of get_operator_to_input_mappings", attrs);
      },
  });
}


std::vector<OperatorSpaceToParallelTensorSpaceMapping> 
  get_operator_to_ptensor_mappings_for_role(ComputationGraphOpAttrs const &attrs,
                                            std::vector<ParallelTensorDimDegrees> const &inputs_degrees,
                                            TensorRole role) {
  switch (role) {
    case TensorRole::INPUT:
      return get_operator_to_input_mappings(attrs, inputs_degrees);
    case TensorRole::WEIGHT:
      return get_operator_to_weight_mappings(attrs, inputs_degrees);
    case TensorRole::OUTPUT:
      return get_operator_to_weight_mappings(attrs, inputs_degrees);
    default:
      PANIC("Unhandled TensorRole", role);
  }
}


} // namespace FlexFlow
