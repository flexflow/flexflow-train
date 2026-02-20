#include "substitutions/substitution.h"
#include "pcg/parallel_computation_graph/parallel_computation_graph_builder.h"
#include "substitutions/open_parallel_tensor_guid_t.h"
#include "substitutions/operator_pattern/operator_attribute_constraint.h"
#include "substitutions/output_graph/output_graph_expr_node.dtg.h"
#include "substitutions/output_graph/output_operator_attrs_assignment.h"
#include "substitutions/pcg_pattern.h"
#include "substitutions/pcg_pattern_builder.h"
#include "substitutions/sub_parallel_computation_graph.h"
#include "substitutions/substitution_builder.h"
#include "substitutions/tensor_pattern/tensor_attribute_pattern.h"
#include "utils/containers/get_only.h"
#include "utils/containers/require_only_key.h"
#include "utils/graph/instances/unordered_set_labelled_open_dataflow_graph.h"
#include "utils/graph/instances/unordered_set_labelled_open_kwarg_dataflow_graph.h"
#include "utils/graph/labelled_open_dataflow_graph/algorithms/get_graph_data.h"
#include "utils/graph/open_dataflow_graph/algorithms/are_isomorphic.h"
#include "utils/integer_conversions.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("is_isomorphic_to(Substitution, Substitution)") {
    auto make_substitution = [] {
      SubstitutionBuilder b;

      auto pair_input = b.add_input(tensor_attribute_pattern_match_all());
      PatternValue p_input = pair_input.first;
      OutputGraphExprValue o_input = pair_input.second;

      auto pair_weight = b.add_input(tensor_attribute_pattern_match_all());
      PatternValue p_weight = pair_weight.first;
      OutputGraphExprValue o_weight = pair_weight.second;

      PatternValue p_mm_output = [&] {
        auto pattern = OperatorAttributePattern{{
            op_type_equals_constraint(OperatorType::LINEAR),
            op_attr_key_equals(OperatorAttributeKey::ACTIVATION,
                               OperatorAttributeValue{
                                   std::optional<Activation>{std::nullopt}}),
        }};

        return require_only_key(
            b.add_pattern_node(
                /*node_pattern=*/pattern,
                /*inputs=*/
                {
                    {
                        TensorSlotName::INPUT,
                        p_input,
                    },
                    {
                        TensorSlotName::WEIGHT,
                        p_weight,
                    },
                },
                /*output_patterns=*/
                {
                    {
                        TensorSlotName::OUTPUT,
                        tensor_attribute_pattern_match_all(),
                    },
                },
                /*name=*/"mm"),
            TensorSlotName::OUTPUT);
      }();

      PatternValue p_relu_output = [&] {
        auto pattern = OperatorAttributePattern{{
            op_type_equals_constraint(OperatorType::RELU),
        }};

        return require_only_key(
            b.add_pattern_node(
                /*node_pattern=*/pattern,
                /*inputs=*/
                {
                    {
                        TensorSlotName::INPUT,
                        p_mm_output,
                    },
                },
                /*output_patterns=*/
                {
                    {
                        TensorSlotName::OUTPUT,
                        tensor_attribute_pattern_match_all(),
                    },
                },
                /*name=*/"relu"),
            TensorSlotName::OUTPUT);
      }();

      OutputGraphExprValue o_fused_output = [&] {
        auto node_expr = OutputOperatorAttrsAssignment{
            b.pattern_node_named("mm"),
            {
                set_attr_to_constant(OperatorAttributeKey::ACTIVATION,
                                     OperatorAttributeValue{Activation::RELU}),
            }};

        return require_only_key(b.add_output_graph_node(
                                    /*node_expr=*/node_expr,
                                    /*input=*/
                                    {
                                        {
                                            TensorSlotName::INPUT,
                                            o_input,
                                        },
                                        {
                                            TensorSlotName::WEIGHT,
                                            o_weight,
                                        },
                                    },
                                    /*output_slots=*/
                                    {
                                        TensorSlotName::OUTPUT,
                                    }),
                                TensorSlotName::OUTPUT);
      }();

      b.equate_outputs(p_relu_output, o_fused_output);

      return b.get_substitution();
    };

    Substitution sub1 = make_substitution();
    Substitution sub2 = make_substitution();

    CHECK(is_isomorphic_to(sub1, sub1));
    CHECK(is_isomorphic_to(sub1, sub2));
  }

  TEST_CASE("is_valid_substitution") {
    SubstitutionBuilder b;

    auto pair_input = b.add_input(tensor_attribute_pattern_match_all());
    PatternValue p_input = pair_input.first;
    OutputGraphExprValue o_input = pair_input.second;

    auto pair_weight = b.add_input(tensor_attribute_pattern_match_all());
    PatternValue p_weight = pair_weight.first;
    OutputGraphExprValue o_weight = pair_weight.second;

    PatternValue p_mm_output = [&] {
      auto pattern = OperatorAttributePattern{{
          op_type_equals_constraint(OperatorType::LINEAR),
          op_attr_key_equals(
              OperatorAttributeKey::ACTIVATION,
              OperatorAttributeValue{std::optional<Activation>{std::nullopt}}),
      }};

      return require_only_key(b.add_pattern_node(
                                  /*node_pattern=*/pattern,
                                  /*inputs=*/
                                  {
                                      {
                                          TensorSlotName::INPUT,
                                          p_input,
                                      },
                                      {
                                          TensorSlotName::WEIGHT,
                                          p_weight,
                                      },
                                  },
                                  /*output_patterns=*/
                                  {
                                      {
                                          TensorSlotName::OUTPUT,
                                          tensor_attribute_pattern_match_all(),
                                      },
                                  },
                                  /*name=*/"mm"),
                              TensorSlotName::OUTPUT);
    }();

    PatternValue p_relu_output = [&] {
      auto pattern = OperatorAttributePattern{{
          op_type_equals_constraint(OperatorType::RELU),
      }};

      return require_only_key(b.add_pattern_node(
                                  /*node_pattern=*/pattern,
                                  /*inputs=*/
                                  {
                                      {
                                          TensorSlotName::INPUT,
                                          p_mm_output,
                                      },
                                  },
                                  /*output_patterns=*/
                                  {
                                      {
                                          TensorSlotName::OUTPUT,
                                          tensor_attribute_pattern_match_all(),
                                      },
                                  },
                                  /*name=*/"relu"),
                              TensorSlotName::OUTPUT);
    }();

    OutputGraphExprValue o_fused_output = [&] {
      auto node_expr = OutputOperatorAttrsAssignment{
          b.pattern_node_named("mm"),
          {
              set_attr_to_constant(OperatorAttributeKey::ACTIVATION,
                                   OperatorAttributeValue{Activation::RELU}),
          }};

      return require_only_key(b.add_output_graph_node(
                                  /*node_expr=*/node_expr,
                                  /*inputs=*/
                                  {
                                      {
                                          TensorSlotName::INPUT,
                                          o_input,
                                      },
                                      {
                                          TensorSlotName::OUTPUT,
                                          o_weight,
                                      },
                                  },
                                  /*output_slots=*/{TensorSlotName::OUTPUT}),
                              TensorSlotName::OUTPUT);
    }();

    b.equate_outputs(p_relu_output, o_fused_output);

    SUBCASE("pattern inputs != mapped inputs") {
      Substitution sub = b.get_substitution();
      sub.pcg_pattern.raw_graph.add_input(13,
                                          tensor_attribute_pattern_match_all());
      CHECK_FALSE(is_valid_substitution(sub));
    }

    SUBCASE("output graph inputs != mapped inputs") {
      Substitution sub = b.get_substitution();
      sub.output_graph_expr.raw_graph.add_input(0, std::monostate{});
      CHECK_FALSE(is_valid_substitution(sub));
    }

    SUBCASE("pattern has no nodes") {
      // Could revamp this test to only trigger the
      // get_nodes(sub.pcg_pattern).empty() case
      Substitution sub = b.get_substitution();
      LabelledOpenKwargDataflowGraph<OperatorAttributePattern,
                                     TensorAttributePattern,
                                     int,
                                     TensorSlotName>
          zero_node_pattern =
              LabelledOpenKwargDataflowGraph<OperatorAttributePattern,
                                             TensorAttributePattern,
                                             int,
                                             TensorSlotName>::
                  create<UnorderedSetLabelledOpenKwargDataflowGraph<
                      OperatorAttributePattern,
                      TensorAttributePattern,
                      int,
                      TensorSlotName>>();
      sub.pcg_pattern = PCGPattern{zero_node_pattern};
      CHECK_FALSE(is_valid_substitution(sub));
    }

    SUBCASE("output graph has no nodes") {
      // Could revamp this test to only trigger the
      // get_nodes(sub.output_graph_expr).empty() case
      Substitution sub = b.get_substitution();
      LabelledOpenKwargDataflowGraph<OutputOperatorAttrsAssignment,
                                     std::monostate,
                                     int,
                                     TensorSlotName>
          zero_node_pattern =
              LabelledOpenKwargDataflowGraph<OutputOperatorAttrsAssignment,
                                             std::monostate,
                                             int,
                                             TensorSlotName>::
                  create<UnorderedSetLabelledOpenKwargDataflowGraph<
                      OutputOperatorAttrsAssignment,
                      std::monostate,
                      int,
                      TensorSlotName>>();
      sub.output_graph_expr = OutputGraphExpr{zero_node_pattern};
      CHECK_FALSE(is_valid_substitution(sub));
    }

    SUBCASE("valid substitution") {
      Substitution sub = b.get_substitution();
      CHECK(is_valid_substitution(sub));
    }
  }
}
