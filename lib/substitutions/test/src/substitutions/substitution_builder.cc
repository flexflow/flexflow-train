#include "substitutions/substitution_builder.h"
#include "substitutions/operator_pattern/operator_attribute_constraint.h"
#include "substitutions/output_graph/output_graph_expr_node.dtg.h"
#include "substitutions/output_graph/output_operator_attrs_assignment.h"
#include "substitutions/substitution.h"
#include "substitutions/tensor_pattern/tensor_attribute_pattern.h"
#include "utils/containers/get_only.h"
#include "utils/graph/instances/unordered_set_labelled_open_dataflow_graph.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("SubstitutionBuilder") {
    OperatorAttributePattern relu_pattern = OperatorAttributePattern{{
        op_type_equals_constraint(OperatorType::RELU),
    }};

    OperatorAttributePattern mm_pattern = OperatorAttributePattern{{
        op_type_equals_constraint(OperatorType::LINEAR),
        op_attr_key_equals(
            OperatorAttributeKey::ACTIVATION,
            OperatorAttributeValue{std::optional<Activation>{std::nullopt}}),
    }};

    std::unordered_map<OperatorAttributeKey, OutputOperatorAttributeExpr>
        fused_mm_relu_attr_assignments = {
            set_attr_to_constant(OperatorAttributeKey::ACTIVATION,
                                 OperatorAttributeValue{Activation::RELU}),
        };

    Substitution correct = [&] {
      auto pattern_g = LabelledOpenDataflowGraph<OperatorAttributePattern,
                                                 TensorAttributePattern>::
          create<
              UnorderedSetLabelledOpenDataflowGraph<OperatorAttributePattern,
                                                    TensorAttributePattern>>();

      PatternInput pattern_i_activation = PatternInput{
          pattern_g.add_input(tensor_attribute_pattern_match_all())};
      PatternInput pattern_i_weights = PatternInput{
          pattern_g.add_input(tensor_attribute_pattern_match_all())};

      NodeAddedResult mm_added = pattern_g.add_node(
          mm_pattern,
          {OpenDataflowValue{pattern_i_activation.raw_dataflow_graph_input},
           OpenDataflowValue{pattern_i_weights.raw_dataflow_graph_input}},
          {tensor_attribute_pattern_match_all()});
      PatternNode pattern_mm_node = PatternNode{mm_added.node};
      DataflowOutput mm_output = get_only(mm_added.outputs);

      NodeAddedResult relu_added =
          pattern_g.add_node(relu_pattern,
                             {OpenDataflowValue{mm_output}},
                             {tensor_attribute_pattern_match_all()});
      PatternNode pattern_relu_node = PatternNode{relu_added.node};
      DataflowOutput relu_output = get_only(relu_added.outputs);

      LabelledOpenDataflowGraph<OutputOperatorAttrsAssignment, std::monostate>
          output_g = LabelledOpenDataflowGraph<OutputOperatorAttrsAssignment,
                                               std::monostate>::
              create<UnorderedSetLabelledOpenDataflowGraph<
                  OutputOperatorAttrsAssignment,
                  std::monostate>>();

      OutputGraphExprInput output_i_activation =
          OutputGraphExprInput{output_g.add_input({})};
      OutputGraphExprInput output_i_weights =
          OutputGraphExprInput{output_g.add_input({})};

      OutputOperatorAttrsAssignment fused_mm_relu_attrs_assignment =
          OutputOperatorAttrsAssignment{
              pattern_mm_node,
              fused_mm_relu_attr_assignments,
          };
      NodeAddedResult fused_mm_relu_added = output_g.add_node(
          fused_mm_relu_attrs_assignment,
          {OpenDataflowValue{output_i_activation.raw_dataflow_graph_input},
           OpenDataflowValue{output_i_weights.raw_dataflow_graph_input}},
          {{}});
      OutputGraphExprNode fused_mm_relu_node =
          OutputGraphExprNode{fused_mm_relu_added.node};
      DataflowOutput fused_mm_relu_output =
          get_only(fused_mm_relu_added.outputs);

      return Substitution{
          PCGPattern{pattern_g},
          OutputGraphExpr{output_g},
          bidict<PatternInput, OutputGraphExprInput>{
              {
                  pattern_i_activation,
                  output_i_activation,
              },
              {
                  pattern_i_weights,
                  output_i_weights,
              },
          },
          bidict<PatternNodeOutput, OutputGraphExprNodeOutput>{
              {
                  PatternNodeOutput{relu_output},
                  OutputGraphExprNodeOutput{fused_mm_relu_output},
              },
          },
      };
    }();

    Substitution result = [&] {
      SubstitutionBuilder b;

      auto [p_input, o_input] =
          b.add_input(tensor_attribute_pattern_match_all());
      auto [p_weight, o_weight] =
          b.add_input(tensor_attribute_pattern_match_all());

      PatternValue p_mm_output =
          get_only(b.add_pattern_node(mm_pattern,
                                      {p_input, p_weight},
                                      {tensor_attribute_pattern_match_all()},
                                      "mm"));

      PatternValue p_relu_output =
          get_only(b.add_pattern_node(relu_pattern,
                                      {p_mm_output},
                                      {tensor_attribute_pattern_match_all()},
                                      "relu"));

      OutputOperatorAttrsAssignment fused_mm_relu_attrs_assignment =
          OutputOperatorAttrsAssignment{
              b.pattern_node_named("mm"),
              fused_mm_relu_attr_assignments,
          };
      OutputGraphExprValue o_fused_output =
          get_only(b.add_output_graph_node(fused_mm_relu_attrs_assignment,
                                           {o_input, o_weight},
                                           nonnegative_int{1}));

      b.equate_outputs(p_relu_output, o_fused_output);

      return b.get_substitution();
    }();

    CHECK(is_isomorphic_to(result, correct));
  }
}
