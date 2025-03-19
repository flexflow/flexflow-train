#include "substitutions/apply_substitution/apply_substitution.h"
#include "pcg/parallel_computation_graph/parallel_computation_graph.h"
#include "pcg/parallel_computation_graph/parallel_computation_graph_builder.h"
#include "substitutions/operator_pattern/operator_attribute_constraint.h"
#include "substitutions/output_graph/output_operator_attrs_assignment.h"
#include "substitutions/sub_parallel_computation_graph.h"
#include "substitutions/substitution_builder.h"
#include "substitutions/tensor_pattern/tensor_attribute_pattern.h"
#include "utils/containers/get_only.h"
#include "utils/integer_conversions.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("apply_substitution") {
    SubstitutionBuilder b;

    auto [p_input, o_input] =
        b.add_input(tensor_attribute_pattern_match_all(), "input");
    auto [p_weight, o_weight] =
        b.add_input(tensor_attribute_pattern_match_all(), "weight");

    PatternValue p_mm_output = [&] {
      auto pattern = OperatorAttributePattern{{
          op_type_equals_constraint(OperatorType::LINEAR),
          op_attr_key_equals(
              OperatorAttributeKey::ACTIVATION,
              OperatorAttributeValue{std::optional<Activation>{std::nullopt}}),
      }};

      return get_only(b.add_pattern_node(pattern,
                                         {p_input, p_weight},
                                         {tensor_attribute_pattern_match_all()},
                                         "mm"));
    }();

    PatternValue p_relu_output = [&] {
      auto pattern = OperatorAttributePattern{{
          op_type_equals_constraint(OperatorType::RELU),
      }};

      return get_only(b.add_pattern_node(pattern,
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
          b.add_output_graph_node(node_expr, {o_input, o_weight}, 1_n));
    }();

    b.equate_outputs(p_relu_output, o_fused_output);

    Substitution sub = b.get_substitution();

    nonnegative_int in_channels = 24_n;
    nonnegative_int batch_size = 4_n;
    nonnegative_int batch_degree = 2_n;
    std::string mm_match = "mm_match";
    std::string relu_match = "relu_match";

    TensorShape input_shape = TensorShape{
        TensorDims{
            FFOrdered<nonnegative_int>{
                batch_size,
                in_channels,
            },
        },
        DataType::FLOAT,
    };

    SubParallelComputationGraph pcg = [&] {
      ParallelComputationGraphBuilder b;

      parallel_tensor_guid_t t = b.create_input_tensor(input_shape);
      t = b.parallel_partition(t, ff_dim_t{0_n}, batch_degree);
      t = b.dense(t,
                  /*outDim=*/16_n,
                  /*activation=*/std::nullopt);
      t = b.gelu(t);
      t = b.dense(t,
                  /*outDim=*/12_n,
                  /*activation=*/std::nullopt,
                  /*use_bias=*/false,
                  /*data_type=*/DataType::FLOAT,
                  /*kernel_initializer=*/std::nullopt,
                  /*bias_initializer=*/std::nullopt,
                  /*name=*/mm_match);
      t = b.relu(t,
                 /*name=*/relu_match);
      t = b.dense(t,
                  /*outDim=*/8_n,
                  /*activation=*/Activation::RELU);

      return sub_pcg_from_full_pcg(b.pcg);
    }();

    PCGPatternMatch match = [&] {
      parallel_layer_guid_t mm_match_layer =
          get_parallel_layer_by_name(pcg, mm_match);
      parallel_layer_guid_t relu_match_layer =
          get_parallel_layer_by_name(pcg, relu_match);
      open_parallel_tensor_guid_t mm_match_layer_input_activations =
          get_layer_inputs(pcg, mm_match_layer).at(0);
      open_parallel_tensor_guid_t mm_match_layer_input_weights =
          get_layer_inputs(pcg, mm_match_layer).at(1);

      return PCGPatternMatch{
          bidict<PatternNode, parallel_layer_guid_t>{
              {b.pattern_node_named("mm"), mm_match_layer},
              {b.pattern_node_named("relu"), relu_match_layer},
          },
          std::unordered_map<PatternInput, open_parallel_tensor_guid_t>{
              {
                  b.pattern_input_named("input"),
                  mm_match_layer_input_activations,
              },
              {
                  b.pattern_input_named("weight"),
                  mm_match_layer_input_weights,
              }},
      };
    }();

    SubParallelComputationGraph result = apply_substitution(pcg, sub, match);

    SubParallelComputationGraph correct = [&] {
      ParallelComputationGraphBuilder b;
      parallel_tensor_guid_t t = b.create_input_tensor(input_shape);
      t = b.parallel_partition(t, ff_dim_t{0_n}, batch_degree);
      t = b.dense(t,
                  /*outDim=*/16_n,
                  /*activation=*/std::nullopt);
      t = b.gelu(t);
      t = b.dense(t,
                  /*outDim=*/12_n,
                  /*activation=*/Activation::RELU,
                  /*use_bias=*/false,
                  /*data_type=*/DataType::FLOAT,
                  /*kernel_initializer=*/std::nullopt,
                  /*bias_initializer=*/std::nullopt,
                  /*name=*/std::nullopt);
      t = b.dense(t,
                  /*outDim=*/8_n,
                  /*activation=*/Activation::RELU);

      return sub_pcg_from_full_pcg(b.pcg);
    }();

    // since the new nodes produced by the substitution have new ids, it's
    // easier/more correct to check that the graphs are isomorphic rather than
    // checking their exact graph data
    CHECK(sub_pcgs_are_isomorphic(result, correct));
  }
}
