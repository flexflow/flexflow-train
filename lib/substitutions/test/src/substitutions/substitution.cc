#include "substitutions/substitution.h"
#include "pcg/parallel_computation_graph/parallel_computation_graph_builder.h"
#include "substitutions/open_parallel_tensor_guid_t.h"
#include "substitutions/operator_pattern/operator_attribute_constraint.h"
#include "substitutions/output_graph/output_graph_expr_node.dtg.h"
#include "substitutions/output_graph/output_operator_attrs_assignment.h"
#include "substitutions/pcg_pattern_builder.h"
#include "substitutions/sub_parallel_computation_graph.h"
#include "substitutions/substitution_builder.h"
#include "substitutions/tensor_pattern/tensor_attribute_pattern.h"
#include "utils/containers/get_only.h"
#include "utils/graph/instances/unordered_set_labelled_open_dataflow_graph.h"
#include "utils/graph/labelled_open_dataflow_graph/algorithms/get_graph_data.h"
#include "utils/graph/open_dataflow_graph/algorithms/are_isomorphic.h"
#include "utils/integer_conversions.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("is_isomorphic_to(Substitution, Substitution)") {
    auto make_substitution = [] {
      SubstitutionBuilder b;

      auto [p_input, o_input] =
          b.add_input(tensor_attribute_pattern_match_all());
      auto [p_weight, o_weight] =
          b.add_input(tensor_attribute_pattern_match_all());

      PatternValue p_mm_output = [&] {
        auto pattern = OperatorAttributePattern{{
            op_type_equals_constraint(OperatorType::LINEAR),
            op_attr_key_equals(OperatorAttributeKey::ACTIVATION,
                               OperatorAttributeValue{
                                   std::optional<Activation>{std::nullopt}}),
        }};

        return get_only(
            b.add_pattern_node(pattern,
                               {p_input, p_weight},
                               {tensor_attribute_pattern_match_all()},
                               "mm"));
      }();

      PatternValue p_relu_output = [&] {
        auto pattern = OperatorAttributePattern{{
            op_type_equals_constraint(OperatorType::RELU),
        }};

        return get_only(
            b.add_pattern_node(pattern,
                               {p_mm_output},
                               {tensor_attribute_pattern_match_all()},
                               "relu"));
      }();

      OutputGraphExprValue o_fused_output = [&] {
        auto node_expr = OutputOperatorAttrsAssignment{
            b.pattern_node_named("mm"),
            {
                set_attr_to_constant(OperatorAttributeKey::ACTIVATION,
                                     OperatorAttributeValue{Activation::RELU}),
            }};

        return get_only(
            b.add_output_graph_node(node_expr, {o_input, o_weight}, 1));
      }();

      b.equate_outputs(p_relu_output, o_fused_output);

      return b.get_substitution();
    };

    Substitution sub1 = make_substitution();
    Substitution sub2 = make_substitution();

    CHECK(is_isomorphic_to(sub1, sub1));
    CHECK(is_isomorphic_to(sub1, sub2));
  }

  // TEST_CASE("is_valid_substitution") {
  //   FAIL("TODO");
  // }
}
