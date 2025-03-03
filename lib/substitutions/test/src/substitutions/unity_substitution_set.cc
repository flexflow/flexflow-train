#include "substitutions/unity_substitution_set.h"
#include "op-attrs/computation_graph_op_attrs.h"
#include "op-attrs/operator_type.h"
#include "op-attrs/ops/attention.h"
#include "op-attrs/ops/combine.h"
#include "op-attrs/ops/conv_2d.h"
#include "op-attrs/ops/element_binary.h"
#include "op-attrs/ops/element_unary.h"
#include "op-attrs/ops/linear.h"
#include "op-attrs/ops/repartition.h"
#include "op-attrs/ops/replicate.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "pcg/parallel_computation_graph/parallel_computation_graph.h"
#include "pcg/parallel_computation_graph/parallel_computation_graph_builder.h"
#include "substitutions/apply_substitution/apply_substitution.h"
#include "substitutions/open_parallel_tensor_guid_t.h"
#include "substitutions/pcg_pattern.h"
#include "substitutions/sub_parallel_computation_graph.h"
#include "substitutions/substitution_builder.h"
#include "utils/containers/get_only.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

template <typename T>
static ParallelLayerAttrs make_layer_attrs(
    T const &op_attrs,
    std::optional<std::string> const &maybe_name = std::nullopt) {
  return ParallelLayerAttrs{
      /*op_attrs=*/PCGOperatorAttrs{op_attrs},
      /*name=*/maybe_name,
  };
};

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("get_substitution_set") {
    MachineSpecification machine_spec = MachineSpecification{
        /*num_nodes=*/2_n,
        /*num_cpus_per_node=*/8_n,
        /*num_gpus_per_node=*/4_n,
        /*inter_node_bandwidth=*/0.0,
        /*intra_node_bandwidth=*/0.0,
    };

    std::vector<Substitution> result = get_substitution_set(machine_spec);

    CHECK(result.size() == 184);
  }

  TEST_CASE("create_replicate_linear_combine") {
    nonnegative_int num_dims = 1_n;
    nonnegative_int degree = 1_n;
    std::string linear_match = "linear_match";

    SUBCASE("use_bias = false") {
      Substitution sub =
          create_replicate_linear_combine(num_dims, degree, false);

      TensorShape input_shape = TensorShape{
          TensorDims{
              FFOrdered<nonnegative_int>{
                  10_n,
                  12_n,
              },
          },
          DataType::FLOAT,
      };

      LinearAttrs linear_attrs = LinearAttrs{
          /*out_channels=*/12_n,
          /*use_bias=*/false,
          /*data_type=*/DataType::FLOAT,
          /*activation=*/std::nullopt,
          /*regularizer=*/std::nullopt,
      };

      ReplicateAttrs replicate_input_attrs = ReplicateAttrs{
          /*replicate_degree=*/degree,
      };

      WeightAttrs projection_weight_attrs = WeightAttrs{
          /*tensor_shape=*/throw_if_unexpected(
              get_projection_shape(linear_attrs, input_shape)),
          /*initializer=*/InitializerAttrs{ZeroInitializerAttrs{}},
      };

      RepartitionAttrs partition_projection_attrs = RepartitionAttrs{
          /*repartition_dim=*/ff_dim_t{1_n},
          /*repartition_degree=*/degree,
      };

      CombineAttrs combine_op_attrs = CombineAttrs{
          /*combine_dim=*/ff_dim_t{
              nonnegative_int{num_dims.unwrap_nonnegative() - 1}},
          /*combine_degree=*/degree,
      };

      SubParallelComputationGraph original_pcg = [&] {
        ParallelComputationGraph pcg = empty_parallel_computation_graph();

        ParallelLayerAddedResult input_added =
            pcg_add_input_layer(pcg, input_shape);

        parallel_tensor_guid_t t_input = get_only(input_added.outputs);

        ParallelLayerAddedResult projection_weight_added = add_parallel_layer(
            pcg, make_layer_attrs(projection_weight_attrs), {}, {});
        parallel_tensor_guid_t t_projection_weight =
            get_only(projection_weight_added.outputs);

        ParallelLayerAddedResult linear_added =
            add_parallel_layer(pcg,
                               make_layer_attrs(linear_attrs, linear_match),
                               {t_input},
                               {t_projection_weight});

        return sub_pcg_from_full_pcg(pcg);
      }();

      PCGPatternMatch match = [&] {
        parallel_layer_guid_t match_layer =
            get_parallel_layer_by_name(original_pcg, linear_match);
        open_parallel_tensor_guid_t match_layer_input_activations =
            get_layer_inputs(original_pcg, match_layer).at(0);
        open_parallel_tensor_guid_t match_layer_input_weights =
            get_layer_inputs(original_pcg, match_layer).at(1);

        return PCGPatternMatch{
            bidict<PatternNode, parallel_layer_guid_t>{
                {PatternNode{Node{0}}, match_layer},
            },
            std::unordered_map<PatternInput, open_parallel_tensor_guid_t>{
                {
                    PatternInput{DataflowGraphInput{0}},
                    match_layer_input_activations,
                },
                {
                    PatternInput{DataflowGraphInput{2}},
                    match_layer_input_weights,
                }},
        };
      }();

      SubParallelComputationGraph result =
          apply_substitution(original_pcg, sub, match);

      SubParallelComputationGraph correct = [&] {
        ParallelComputationGraph pcg = empty_parallel_computation_graph();

        ParallelLayerAddedResult input_added =
            pcg_add_input_layer(pcg, input_shape);

        parallel_tensor_guid_t t_input = get_only(input_added.outputs);

        ParallelLayerAddedResult replicate_input_added = add_parallel_layer(
            pcg, make_layer_attrs(replicate_input_attrs), {t_input}, {});
        parallel_tensor_guid_t t_replicated_input =
            get_only(replicate_input_added.outputs);

        ParallelLayerAddedResult projection_weight_added = add_parallel_layer(
            pcg, make_layer_attrs(projection_weight_attrs), {}, {});
        parallel_tensor_guid_t t_projection_weight =
            get_only(projection_weight_added.outputs);

        ParallelLayerAddedResult partition_projection_added =
            add_parallel_layer(pcg,
                               make_layer_attrs(partition_projection_attrs),
                               {t_projection_weight},
                               {});
        parallel_tensor_guid_t t_partitioned_projection_weight =
            get_only(partition_projection_added.outputs);

        ParallelLayerAddedResult replicate_linear_added =
            add_parallel_layer(pcg,
                               make_layer_attrs(linear_attrs),
                               {t_replicated_input},
                               {t_partitioned_projection_weight});
        parallel_tensor_guid_t t_replicated_linear =
            get_only(replicate_linear_added.outputs);

        ParallelLayerAddedResult combine_added = add_parallel_layer(
            pcg, make_layer_attrs(combine_op_attrs), {t_replicated_linear}, {});
        parallel_tensor_guid_t t_combine = get_only(combine_added.outputs);

        return sub_pcg_from_full_pcg(pcg);
      }();

      CHECK(sub_pcgs_are_isomorphic(result, correct));
    }
  }

  TEST_CASE("create_partition_linear_combine") {
    nonnegative_int num_dims = 1_n;
    nonnegative_int degree = 2_n;
    std::string linear_match = "linear_match";

    SUBCASE("use_bias = false") {
      Substitution sub =
          create_partition_linear_combine(num_dims, degree, false);

      TensorShape input_shape = TensorShape{
          TensorDims{
              FFOrdered<nonnegative_int>{
                  10_n,
                  12_n,
              },
          },
          DataType::FLOAT,
      };

      LinearAttrs linear_attrs = LinearAttrs{
          /*out_channels=*/12_n,
          /*use_bias=*/false,
          /*data_type=*/DataType::FLOAT,
          /*activation=*/std::nullopt,
          /*regularizer=*/std::nullopt,
      };

      RepartitionAttrs partition_input_attrs = RepartitionAttrs{
          /*repartition_dim=*/ff_dim_t{0_n},
          /*repartition_degree=*/degree,
      };

      WeightAttrs projection_weight_attrs = WeightAttrs{
          /*tensor_shape=*/throw_if_unexpected(
              get_projection_shape(linear_attrs, input_shape)),
          /*initializer=*/InitializerAttrs{ZeroInitializerAttrs{}},
      };

      ReplicateAttrs replicate_projection_attrs = ReplicateAttrs{
          /*replicate_degree=*/degree,
      };

      CombineAttrs combine_op_attrs = CombineAttrs{
          /*combine_dim=*/ff_dim_t{
              nonnegative_int{num_dims.unwrap_nonnegative() - 1}},
          /*combine_degree=*/degree,
      };

      SubParallelComputationGraph original_pcg = [&] {
        ParallelComputationGraph pcg = empty_parallel_computation_graph();

        ParallelLayerAddedResult input_added =
            pcg_add_input_layer(pcg, input_shape);

        parallel_tensor_guid_t t_input = get_only(input_added.outputs);

        ParallelLayerAddedResult projection_weight_added = add_parallel_layer(
            pcg, make_layer_attrs(projection_weight_attrs), {}, {});
        parallel_tensor_guid_t t_projection_weight =
            get_only(projection_weight_added.outputs);

        ParallelLayerAddedResult linear_added =
            add_parallel_layer(pcg,
                               make_layer_attrs(linear_attrs, linear_match),
                               {t_input},
                               {t_projection_weight});

        return sub_pcg_from_full_pcg(pcg);
      }();

      PCGPatternMatch match = [&] {
        parallel_layer_guid_t match_layer =
            get_parallel_layer_by_name(original_pcg, linear_match);
        open_parallel_tensor_guid_t match_layer_input_activations =
            get_layer_inputs(original_pcg, match_layer).at(0);
        open_parallel_tensor_guid_t match_layer_input_weights =
            get_layer_inputs(original_pcg, match_layer).at(1);

        return PCGPatternMatch{
            bidict<PatternNode, parallel_layer_guid_t>{
                {PatternNode{Node{0}}, match_layer},
            },
            std::unordered_map<PatternInput, open_parallel_tensor_guid_t>{
                {
                    PatternInput{DataflowGraphInput{0}},
                    match_layer_input_activations,
                },
                {
                    PatternInput{DataflowGraphInput{2}},
                    match_layer_input_weights,
                }},
        };
      }();

      SubParallelComputationGraph result =
          apply_substitution(original_pcg, sub, match);

      SubParallelComputationGraph correct = [&] {
        ParallelComputationGraph pcg = empty_parallel_computation_graph();

        ParallelLayerAddedResult input_added =
            pcg_add_input_layer(pcg, input_shape);

        parallel_tensor_guid_t t_input = get_only(input_added.outputs);

        ParallelLayerAddedResult partition_input_added = add_parallel_layer(
            pcg, make_layer_attrs(partition_input_attrs), {t_input}, {});
        parallel_tensor_guid_t t_partitioned_input =
            get_only(partition_input_added.outputs);

        ParallelLayerAddedResult projection_weight_added = add_parallel_layer(
            pcg, make_layer_attrs(projection_weight_attrs), {}, {});
        parallel_tensor_guid_t t_projection_weight =
            get_only(projection_weight_added.outputs);

        ParallelLayerAddedResult replicate_projection_added =
            add_parallel_layer(pcg,
                               make_layer_attrs(replicate_projection_attrs),
                               {t_projection_weight},
                               {});
        parallel_tensor_guid_t t_replicated_projection_weight =
            get_only(replicate_projection_added.outputs);

        ParallelLayerAddedResult partition_linear_added =
            add_parallel_layer(pcg,
                               make_layer_attrs(linear_attrs),
                               {t_partitioned_input},
                               {t_replicated_projection_weight});
        parallel_tensor_guid_t t_partitioned_linear =
            get_only(partition_linear_added.outputs);

        ParallelLayerAddedResult combine_added =
            add_parallel_layer(pcg,
                               make_layer_attrs(combine_op_attrs),
                               {t_partitioned_linear},
                               {});
        parallel_tensor_guid_t t_combine = get_only(combine_added.outputs);

        return sub_pcg_from_full_pcg(pcg);
      }();

      CHECK(sub_pcgs_are_isomorphic(result, correct));
    }
  }

  TEST_CASE("create_partition_conv2d_combine") {
    nonnegative_int outChannels = 6_n;
    nonnegative_int kernelH = 5_n;
    nonnegative_int kernelW = 4_n;
    nonnegative_int strideH = 3_n;
    nonnegative_int strideW = 2_n;
    nonnegative_int paddingH = 1_n;
    nonnegative_int paddingW = 0_n;
    nonnegative_int num_dims = 4_n;
    nonnegative_int degree = 1_n;
    std::string conv2d_match = "conv2d_match";

    Substitution sub = create_partition_conv2d_combine(num_dims, degree);

    TensorShape input_shape = TensorShape{
        TensorDims{
            FFOrdered<nonnegative_int>{
                12_n,
                3_n,
                10_n,
                10_n,
            },
        },
        DataType::FLOAT,
    };

    Conv2DAttrs conv2d_attrs = Conv2DAttrs{/*outChannels=*/outChannels,
                                           /*kernelH=*/kernelH,
                                           /*kernelW=*/kernelW,
                                           /*strideH=*/strideH,
                                           /*strideW=*/strideW,
                                           /*paddingH=*/paddingH,
                                           /*paddingW=*/paddingW,
                                           /*groups=*/1_n,
                                           /*activation=*/std::nullopt,
                                           /*use_bias=*/false};

    RepartitionAttrs partition_input_attrs = RepartitionAttrs{
        /*repartition_dim=*/ff_dim_t{0_n},
        /*repartition_degree=*/degree,
    };

    ReplicateAttrs replicate_weight_attrs = ReplicateAttrs{
        /*replicate_degree=*/degree,
    };

    CombineAttrs combine_attrs = CombineAttrs{
        /*combine_dim=*/ff_dim_t{
            nonnegative_int{num_dims.unwrap_nonnegative() - 1}},
        /*combine_degree=*/degree,
    };

    SubParallelComputationGraph original_pcg = [&] {
      ParallelComputationGraph pcg = empty_parallel_computation_graph();

      ParallelLayerAddedResult input_added =
          pcg_add_input_layer(pcg, input_shape);

      parallel_tensor_guid_t t_input = get_only(input_added.outputs);

      TensorShape casted_input_shape =
          get_reduced_shape(get_parallel_tensor_shape(pcg, t_input));

      WeightAttrs projection_weight_attrs = WeightAttrs{
          /*tensor_shape=*/
          get_weight_shapes(conv2d_attrs, casted_input_shape).at(0),
          /*initializer=*/InitializerAttrs{ZeroInitializerAttrs{}},
      };

      ParallelLayerAddedResult projection_weight_added = add_parallel_layer(
          pcg, make_layer_attrs(projection_weight_attrs), {}, {});
      parallel_tensor_guid_t t_projection_weight =
          get_only(projection_weight_added.outputs);

      ParallelLayerAddedResult conv_2d_added =
          add_parallel_layer(pcg,
                             make_layer_attrs(conv2d_attrs, conv2d_match),
                             {t_input},
                             {t_projection_weight});

      return sub_pcg_from_full_pcg(pcg);
    }();

    PCGPatternMatch match = [&] {
      parallel_layer_guid_t match_layer =
          get_parallel_layer_by_name(original_pcg, conv2d_match);
      open_parallel_tensor_guid_t match_layer_input_activations =
          get_layer_inputs(original_pcg, match_layer).at(0);
      open_parallel_tensor_guid_t match_layer_input_weights =
          get_layer_inputs(original_pcg, match_layer).at(1);

      return PCGPatternMatch{
          bidict<PatternNode, parallel_layer_guid_t>{
              {PatternNode{Node{0}}, match_layer},
          },
          std::unordered_map<PatternInput, open_parallel_tensor_guid_t>{
              {
                  PatternInput{DataflowGraphInput{0}},
                  match_layer_input_activations,
              },
              {
                  PatternInput{DataflowGraphInput{2}},
                  match_layer_input_weights,
              }},
      };
    }();

    SubParallelComputationGraph result =
        apply_substitution(original_pcg, sub, match);

    SubParallelComputationGraph correct = [&] {
      ParallelComputationGraph pcg = empty_parallel_computation_graph();

      ParallelLayerAddedResult input_added =
          pcg_add_input_layer(pcg, input_shape);

      parallel_tensor_guid_t t_input = get_only(input_added.outputs);

      ParallelLayerAddedResult partition_input_added = add_parallel_layer(
          pcg, make_layer_attrs(partition_input_attrs), {t_input}, {});
      parallel_tensor_guid_t t_partitioned_input =
          get_only(partition_input_added.outputs);

      TensorShape casted_input_shape =
          get_reduced_shape(get_parallel_tensor_shape(pcg, t_input));

      WeightAttrs weight_attrs = WeightAttrs{
          /*tensor_shape=*/
          get_weight_shapes(conv2d_attrs, casted_input_shape).at(0),
          /*initializer=*/InitializerAttrs{ZeroInitializerAttrs{}},
      };

      ParallelLayerAddedResult weight_added =
          add_parallel_layer(pcg, make_layer_attrs(weight_attrs), {}, {});
      parallel_tensor_guid_t t_weight = get_only(weight_added.outputs);

      ParallelLayerAddedResult replicate_weight_added = add_parallel_layer(
          pcg, make_layer_attrs(replicate_weight_attrs), {t_weight}, {});
      parallel_tensor_guid_t t_replicated_weight =
          get_only(replicate_weight_added.outputs);

      ParallelLayerAddedResult partition_conv2d_added =
          add_parallel_layer(pcg,
                             make_layer_attrs(conv2d_attrs),
                             {t_partitioned_input},
                             {t_replicated_weight});
      parallel_tensor_guid_t t_partitioned_conv2d =
          get_only(partition_conv2d_added.outputs);

      ParallelLayerAddedResult combine_added = add_parallel_layer(
          pcg, make_layer_attrs(combine_attrs), {t_partitioned_conv2d}, {});
      parallel_tensor_guid_t t_combine = get_only(combine_added.outputs);

      return sub_pcg_from_full_pcg(pcg);
    }();

    CHECK(sub_pcgs_are_isomorphic(result, correct));
  }

  TEST_CASE("create_partition_attention_combine") {
    nonnegative_int embed_dim = 8_n;
    nonnegative_int num_heads = 6_n;
    nonnegative_int degree = 1_n;
    std::string attention_match = "attention_match";

    Substitution sub = create_partition_attention_combine(num_heads, degree);

    TensorShape query_shape = TensorShape{
        TensorDims{
            FFOrdered<nonnegative_int>{
                12_n,
                16_n,
                10_n,
            },
        },
        DataType::FLOAT,
    };
    TensorShape key_shape = query_shape;
    TensorShape value_shape = query_shape;

    MultiHeadAttentionAttrs attention_attrs = MultiHeadAttentionAttrs{
        /*embed_dim=*/embed_dim,
        /*num_heads=*/num_heads,
        /*kdim=*/embed_dim,
        /*vdim=*/embed_dim,
        /*dropout=*/0,
        /*bias=*/false,
        /*add_bias_kv=*/false,
        /*add_zero_attn=*/false,
    };

    RepartitionAttrs partition_input_attrs = RepartitionAttrs{
        /*repartition_dim=*/ff_dim_t{0_n},
        /*repartition_degree=*/degree,
    };

    WeightAttrs weight_attrs = WeightAttrs{
        /*tensor_shape=*/
        throw_if_unexpected(get_weights_shape(
            attention_attrs, query_shape, key_shape, value_shape)),
        /*initializer=*/InitializerAttrs{ZeroInitializerAttrs{}},
    };

    ReplicateAttrs replicate_weight_attrs = ReplicateAttrs{
        /*replicate_degree=*/degree,
    };

    CombineAttrs combine_attrs = CombineAttrs{
        /*combine_dim=*/ff_dim_t{2_n},
        /*combine_degree=*/degree,
    };

    SubParallelComputationGraph original_pcg = [&] {
      ParallelComputationGraph pcg = empty_parallel_computation_graph();

      ParallelLayerAddedResult query_added =
          pcg_add_input_layer(pcg, query_shape);
      parallel_tensor_guid_t t_query = get_only(query_added.outputs);

      ParallelLayerAddedResult key_added = pcg_add_input_layer(pcg, key_shape);
      parallel_tensor_guid_t t_key = get_only(key_added.outputs);

      ParallelLayerAddedResult value_added =
          pcg_add_input_layer(pcg, value_shape);
      parallel_tensor_guid_t t_value = get_only(value_added.outputs);

      ParallelLayerAddedResult weight_added =
          add_parallel_layer(pcg, make_layer_attrs(weight_attrs), {}, {});
      parallel_tensor_guid_t t_weight = get_only(weight_added.outputs);

      ParallelLayerAddedResult attention_added =
          add_parallel_layer(pcg,
                             make_layer_attrs(attention_attrs, attention_match),
                             {t_query, t_key, t_value},
                             {t_weight});

      return sub_pcg_from_full_pcg(pcg);
    }();

    PCGPatternMatch match = [&] {
      parallel_layer_guid_t match_layer =
          get_parallel_layer_by_name(original_pcg, attention_match);
      open_parallel_tensor_guid_t match_layer_query =
          get_layer_inputs(original_pcg, match_layer).at(0);
      open_parallel_tensor_guid_t match_layer_key =
          get_layer_inputs(original_pcg, match_layer).at(1);
      open_parallel_tensor_guid_t match_layer_value =
          get_layer_inputs(original_pcg, match_layer).at(2);
      open_parallel_tensor_guid_t match_layer_input_weights =
          get_layer_inputs(original_pcg, match_layer).at(3);

      return PCGPatternMatch{
          bidict<PatternNode, parallel_layer_guid_t>{
              {PatternNode{Node{0}}, match_layer},
          },
          std::unordered_map<PatternInput, open_parallel_tensor_guid_t>{
              {
                  PatternInput{DataflowGraphInput{0}},
                  match_layer_query,
              },
              {
                  PatternInput{DataflowGraphInput{2}},
                  match_layer_key,
              },
              {
                  PatternInput{DataflowGraphInput{4}},
                  match_layer_value,
              },
              {
                  PatternInput{DataflowGraphInput{6}},
                  match_layer_input_weights,
              }},
      };
    }();

    SubParallelComputationGraph result =
        apply_substitution(original_pcg, sub, match);

    SubParallelComputationGraph correct = [&] {
      ParallelComputationGraph pcg = empty_parallel_computation_graph();

      ParallelLayerAddedResult query_added =
          pcg_add_input_layer(pcg, query_shape);
      parallel_tensor_guid_t t_query = get_only(query_added.outputs);

      ParallelLayerAddedResult key_added = pcg_add_input_layer(pcg, key_shape);
      parallel_tensor_guid_t t_key = get_only(key_added.outputs);

      ParallelLayerAddedResult value_added =
          pcg_add_input_layer(pcg, value_shape);
      parallel_tensor_guid_t t_value = get_only(value_added.outputs);

      ParallelLayerAddedResult weight_added =
          add_parallel_layer(pcg, make_layer_attrs(weight_attrs), {}, {});
      parallel_tensor_guid_t t_weight = get_only(weight_added.outputs);

      ParallelLayerAddedResult partition_query_added = add_parallel_layer(
          pcg, make_layer_attrs(partition_input_attrs), {t_query}, {});
      parallel_tensor_guid_t t_partitioned_query =
          get_only(partition_query_added.outputs);

      ParallelLayerAddedResult partition_key_added = add_parallel_layer(
          pcg, make_layer_attrs(partition_input_attrs), {t_key}, {});
      parallel_tensor_guid_t t_partitioned_key =
          get_only(partition_key_added.outputs);

      ParallelLayerAddedResult partition_value_added = add_parallel_layer(
          pcg, make_layer_attrs(partition_input_attrs), {t_value}, {});
      parallel_tensor_guid_t t_partitioned_value =
          get_only(partition_value_added.outputs);

      ParallelLayerAddedResult replicate_weight_added = add_parallel_layer(
          pcg, make_layer_attrs(replicate_weight_attrs), {t_weight}, {});
      parallel_tensor_guid_t t_replicated_weight =
          get_only(replicate_weight_added.outputs);

      ParallelLayerAddedResult partition_attention_added = add_parallel_layer(
          pcg,
          make_layer_attrs(attention_attrs),
          {t_partitioned_query, t_partitioned_key, t_partitioned_value},
          {t_replicated_weight});
      parallel_tensor_guid_t t_partitioned_attention =
          get_only(partition_attention_added.outputs);

      ParallelLayerAddedResult combine_added = add_parallel_layer(
          pcg, make_layer_attrs(combine_attrs), {t_partitioned_attention}, {});
      parallel_tensor_guid_t t_combine = get_only(combine_added.outputs);

      return sub_pcg_from_full_pcg(pcg);
    }();

    CHECK(sub_pcgs_are_isomorphic(result, correct));
  }

  TEST_CASE("create_replicate_attention_reduce") {
    nonnegative_int embed_dim = 8_n;
    nonnegative_int num_heads = 6_n;
    nonnegative_int degree = 1_n;
    std::string attention_match = "attention_match";

    Substitution sub = create_replicate_attention_reduce(num_heads, degree);

    TensorShape query_shape = TensorShape{
        TensorDims{
            FFOrdered<nonnegative_int>{
                12_n,
                16_n,
                10_n,
            },
        },
        DataType::FLOAT,
    };
    TensorShape key_shape = query_shape;
    TensorShape value_shape = query_shape;

    MultiHeadAttentionAttrs attention_attrs = MultiHeadAttentionAttrs{
        /*embed_dim=*/embed_dim,
        /*num_heads=*/num_heads,
        /*kdim=*/embed_dim,
        /*vdim=*/embed_dim,
        /*dropout=*/0,
        /*bias=*/false,
        /*add_bias_kv=*/false,
        /*add_zero_attn=*/false,
    };

    ReplicateAttrs replicate_input_attrs = ReplicateAttrs{
        /*replicate_degree=*/degree,
    };

    WeightAttrs weight_attrs = WeightAttrs{
        /*tensor_shape=*/
        throw_if_unexpected(get_weights_shape(
            attention_attrs, query_shape, key_shape, value_shape)),
        /*initializer=*/InitializerAttrs{ZeroInitializerAttrs{}},
    };

    RepartitionAttrs partition_weight_attrs = RepartitionAttrs{
        /*repartition_dim=*/ff_dim_t{1_n},
        /*repartition_degree=*/degree,
    };

    ReductionAttrs reduction_attrs = ReductionAttrs{
        /*reduction_degree=*/degree,
    };

    SubParallelComputationGraph original_pcg = [&] {
      ParallelComputationGraph pcg = empty_parallel_computation_graph();

      ParallelLayerAddedResult query_added =
          pcg_add_input_layer(pcg, query_shape);
      parallel_tensor_guid_t t_query = get_only(query_added.outputs);

      ParallelLayerAddedResult key_added = pcg_add_input_layer(pcg, key_shape);
      parallel_tensor_guid_t t_key = get_only(key_added.outputs);

      ParallelLayerAddedResult value_added =
          pcg_add_input_layer(pcg, value_shape);
      parallel_tensor_guid_t t_value = get_only(value_added.outputs);

      ParallelLayerAddedResult weight_added =
          add_parallel_layer(pcg, make_layer_attrs(weight_attrs), {}, {});
      parallel_tensor_guid_t t_weight = get_only(weight_added.outputs);

      ParallelLayerAddedResult attention_added =
          add_parallel_layer(pcg,
                             make_layer_attrs(attention_attrs, attention_match),
                             {t_query, t_key, t_value},
                             {t_weight});

      return sub_pcg_from_full_pcg(pcg);
    }();

    PCGPatternMatch match = [&] {
      parallel_layer_guid_t match_layer =
          get_parallel_layer_by_name(original_pcg, attention_match);
      open_parallel_tensor_guid_t match_layer_query =
          get_layer_inputs(original_pcg, match_layer).at(0);
      open_parallel_tensor_guid_t match_layer_key =
          get_layer_inputs(original_pcg, match_layer).at(1);
      open_parallel_tensor_guid_t match_layer_value =
          get_layer_inputs(original_pcg, match_layer).at(2);
      open_parallel_tensor_guid_t match_layer_input_weights =
          get_layer_inputs(original_pcg, match_layer).at(3);

      return PCGPatternMatch{
          bidict<PatternNode, parallel_layer_guid_t>{
              {PatternNode{Node{0}}, match_layer},
          },
          std::unordered_map<PatternInput, open_parallel_tensor_guid_t>{
              {
                  PatternInput{DataflowGraphInput{0}},
                  match_layer_query,
              },
              {
                  PatternInput{DataflowGraphInput{2}},
                  match_layer_key,
              },
              {
                  PatternInput{DataflowGraphInput{4}},
                  match_layer_value,
              },
              {
                  PatternInput{DataflowGraphInput{6}},
                  match_layer_input_weights,
              }},
      };
    }();

    SubParallelComputationGraph result =
        apply_substitution(original_pcg, sub, match);

    SubParallelComputationGraph correct = [&] {
      ParallelComputationGraph pcg = empty_parallel_computation_graph();

      ParallelLayerAddedResult query_added =
          pcg_add_input_layer(pcg, query_shape);
      parallel_tensor_guid_t t_query = get_only(query_added.outputs);

      ParallelLayerAddedResult key_added = pcg_add_input_layer(pcg, key_shape);
      parallel_tensor_guid_t t_key = get_only(key_added.outputs);

      ParallelLayerAddedResult value_added =
          pcg_add_input_layer(pcg, value_shape);
      parallel_tensor_guid_t t_value = get_only(value_added.outputs);

      ParallelLayerAddedResult weight_added =
          add_parallel_layer(pcg, make_layer_attrs(weight_attrs), {}, {});
      parallel_tensor_guid_t t_weight = get_only(weight_added.outputs);

      ParallelLayerAddedResult replicate_query_added = add_parallel_layer(
          pcg, make_layer_attrs(replicate_input_attrs), {t_query}, {});
      parallel_tensor_guid_t t_replicated_query =
          get_only(replicate_query_added.outputs);

      ParallelLayerAddedResult replicate_key_added = add_parallel_layer(
          pcg, make_layer_attrs(replicate_input_attrs), {t_key}, {});
      parallel_tensor_guid_t t_replicated_key =
          get_only(replicate_key_added.outputs);

      ParallelLayerAddedResult replicate_value_added = add_parallel_layer(
          pcg, make_layer_attrs(replicate_input_attrs), {t_value}, {});
      parallel_tensor_guid_t t_replicated_value =
          get_only(replicate_value_added.outputs);

      ParallelLayerAddedResult partition_weight_added = add_parallel_layer(
          pcg, make_layer_attrs(partition_weight_attrs), {t_weight}, {});
      parallel_tensor_guid_t t_partitioned_weight =
          get_only(partition_weight_added.outputs);

      ParallelLayerAddedResult replicate_attention_added = add_parallel_layer(
          pcg,
          make_layer_attrs(attention_attrs),
          {t_replicated_query, t_replicated_key, t_replicated_value},
          {t_partitioned_weight});
      parallel_tensor_guid_t t_replicated_attention =
          get_only(replicate_attention_added.outputs);

      ParallelLayerAddedResult reduce_added = add_parallel_layer(
          pcg, make_layer_attrs(reduction_attrs), {t_replicated_attention}, {});
      parallel_tensor_guid_t t_reduction = get_only(reduce_added.outputs);

      return sub_pcg_from_full_pcg(pcg);
    }();

    CHECK(sub_pcgs_are_isomorphic(result, correct));
  }

  TEST_CASE("create_partition_softmax_combine") {
    nonnegative_int degree = 1_n;
    ff_dim_t softmax_dim = ff_dim_t{1_n};
    ff_dim_t partition_dim = ff_dim_t{0_n};
    std::string softmax_match = "softmax_match";

    Substitution sub =
        create_partition_softmax_combine(softmax_dim, partition_dim, degree);

    TensorShape input_shape = TensorShape{
        TensorDims{
            FFOrdered<nonnegative_int>{
                10_n,
                10_n,
            },
        },
        DataType::FLOAT,
    };

    SoftmaxAttrs softmax_attrs = SoftmaxAttrs{
        /*softmax_dim=*/softmax_dim,
    };

    RepartitionAttrs partition_input_attrs = RepartitionAttrs{
        /*repartition_dim=*/partition_dim,
        /*repartition_degree=*/degree,
    };

    CombineAttrs combine_attrs = CombineAttrs{
        /*combine_dim=*/ff_dim_t{partition_dim},
        /*combine_degree=*/degree,
    };

    SubParallelComputationGraph original_pcg = [&] {
      ParallelComputationGraph pcg = empty_parallel_computation_graph();

      ParallelLayerAddedResult input_added =
          pcg_add_input_layer(pcg, input_shape);

      parallel_tensor_guid_t t_input = get_only(input_added.outputs);

      ParallelLayerAddedResult softmax_added = add_parallel_layer(
          pcg, make_layer_attrs(softmax_attrs, softmax_match), {t_input}, {});

      return sub_pcg_from_full_pcg(pcg);
    }();

    PCGPatternMatch match = [&] {
      parallel_layer_guid_t match_layer =
          get_parallel_layer_by_name(original_pcg, softmax_match);
      open_parallel_tensor_guid_t match_layer_input =
          get_layer_inputs(original_pcg, match_layer).at(0);

      return PCGPatternMatch{
          bidict<PatternNode, parallel_layer_guid_t>{
              {PatternNode{Node{0}}, match_layer},
          },
          std::unordered_map<PatternInput, open_parallel_tensor_guid_t>{{
              PatternInput{DataflowGraphInput{0}},
              match_layer_input,
          }},
      };
    }();

    SubParallelComputationGraph result =
        apply_substitution(original_pcg, sub, match);

    SubParallelComputationGraph correct = [&] {
      ParallelComputationGraph pcg = empty_parallel_computation_graph();

      ParallelLayerAddedResult input_added =
          pcg_add_input_layer(pcg, input_shape);

      parallel_tensor_guid_t t_input = get_only(input_added.outputs);

      ParallelLayerAddedResult partition_input_added = add_parallel_layer(
          pcg, make_layer_attrs(partition_input_attrs), {t_input}, {});
      parallel_tensor_guid_t t_partitioned_input =
          get_only(partition_input_added.outputs);

      ParallelLayerAddedResult partition_softmax_added = add_parallel_layer(
          pcg, make_layer_attrs(softmax_attrs), {t_partitioned_input}, {});
      parallel_tensor_guid_t t_partitioned_softmax =
          get_only(partition_softmax_added.outputs);

      ParallelLayerAddedResult combine_added = add_parallel_layer(
          pcg, make_layer_attrs(combine_attrs), {t_partitioned_softmax}, {});
      parallel_tensor_guid_t t_combine = get_only(combine_added.outputs);

      return sub_pcg_from_full_pcg(pcg);
    }();

    CHECK(sub_pcgs_are_isomorphic(result, correct));
  }

  TEST_CASE("create_partition_add_combine") {
    nonnegative_int degree = 1_n;
    ff_dim_t parallel_dim = ff_dim_t{1_n};
    std::string add_match = "add_match";

    Substitution sub = create_partition_add_combine(parallel_dim, degree);

    TensorShape lhs_shape = TensorShape{
        TensorDims{
            FFOrdered<nonnegative_int>{
                10_n,
                15_n,
            },
        },
        DataType::FLOAT,
    };

    TensorShape rhs_shape = lhs_shape;

    ElementBinaryAttrs add_attrs = ElementBinaryAttrs{
        OperatorType::EW_ADD,
        DataType::FLOAT,
        false,
        false,
    };

    RepartitionAttrs partition_input_attrs = RepartitionAttrs{
        /*repartition_dim=*/parallel_dim,
        /*repartition_degree=*/degree,
    };

    CombineAttrs combine_attrs = CombineAttrs{
        /*combine_dim=*/parallel_dim,
        /*combine_degree=*/degree,
    };

    SubParallelComputationGraph original_pcg = [&] {
      ParallelComputationGraph pcg = empty_parallel_computation_graph();

      ParallelLayerAddedResult lhs_added = pcg_add_input_layer(pcg, lhs_shape);
      parallel_tensor_guid_t t_lhs = get_only(lhs_added.outputs);

      ParallelLayerAddedResult rhs_added = pcg_add_input_layer(pcg, rhs_shape);
      parallel_tensor_guid_t t_rhs = get_only(rhs_added.outputs);

      ParallelLayerAddedResult output_added = add_parallel_layer(
          pcg, make_layer_attrs(add_attrs, add_match), {t_lhs, t_rhs}, {});

      return sub_pcg_from_full_pcg(pcg);
    }();

    PCGPatternMatch match = [&] {
      parallel_layer_guid_t match_layer =
          get_parallel_layer_by_name(original_pcg, add_match);
      open_parallel_tensor_guid_t add_match_layer_lhs =
          get_layer_inputs(original_pcg, match_layer).at(0);
      open_parallel_tensor_guid_t add_match_layer_rhs =
          get_layer_inputs(original_pcg, match_layer).at(1);

      return PCGPatternMatch{
          bidict<PatternNode, parallel_layer_guid_t>{
              {PatternNode{Node{0}}, match_layer},
          },
          std::unordered_map<PatternInput, open_parallel_tensor_guid_t>{
              {
                  PatternInput{DataflowGraphInput{0}},
                  add_match_layer_lhs,
              },
              {
                  PatternInput{DataflowGraphInput{2}},
                  add_match_layer_rhs,
              }},
      };
    }();

    SubParallelComputationGraph result =
        apply_substitution(original_pcg, sub, match);

    SubParallelComputationGraph correct = [&] {
      ParallelComputationGraph pcg = empty_parallel_computation_graph();

      ParallelLayerAddedResult lhs_added = pcg_add_input_layer(pcg, lhs_shape);
      parallel_tensor_guid_t t_lhs = get_only(lhs_added.outputs);

      ParallelLayerAddedResult rhs_added = pcg_add_input_layer(pcg, rhs_shape);
      parallel_tensor_guid_t t_rhs = get_only(rhs_added.outputs);

      ParallelLayerAddedResult partition_lhs_added = add_parallel_layer(
          pcg, make_layer_attrs(partition_input_attrs), {t_lhs}, {});
      parallel_tensor_guid_t t_partitioned_lhs =
          get_only(partition_lhs_added.outputs);

      ParallelLayerAddedResult partition_rhs_added = add_parallel_layer(
          pcg, make_layer_attrs(partition_input_attrs), {t_rhs}, {});
      parallel_tensor_guid_t t_partitioned_rhs =
          get_only(partition_rhs_added.outputs);

      ParallelLayerAddedResult partition_add_added =
          add_parallel_layer(pcg,
                             make_layer_attrs(add_attrs, add_match),
                             {t_partitioned_lhs, t_partitioned_rhs},
                             {});
      parallel_tensor_guid_t t_partitioned_add =
          get_only(partition_add_added.outputs);

      ParallelLayerAddedResult combine_added = add_parallel_layer(
          pcg, make_layer_attrs(combine_attrs), {t_partitioned_add}, {});
      parallel_tensor_guid_t t_combine = get_only(combine_added.outputs);

      return sub_pcg_from_full_pcg(pcg);
    }();

    CHECK(sub_pcgs_are_isomorphic(result, correct));
  }

  TEST_CASE("create_partition_relu_combine") {
    nonnegative_int degree = 1_n;
    ff_dim_t parallel_dim = ff_dim_t{1_n};
    std::string relu_match = "relu_match";

    Substitution sub = create_partition_relu_combine(parallel_dim, degree);

    TensorShape input_shape = TensorShape{
        TensorDims{
            FFOrdered<nonnegative_int>{
                10_n,
                10_n,
            },
        },
        DataType::FLOAT,
    };

    ElementUnaryAttrs relu_attrs = ElementUnaryAttrs{
        OperatorType::RELU,
        std::nullopt,
    };

    RepartitionAttrs partition_input_attrs = RepartitionAttrs{
        /*repartition_dim=*/parallel_dim,
        /*repartition_degree=*/degree,
    };

    CombineAttrs combine_attrs = CombineAttrs{
        /*combine_dim=*/ff_dim_t{parallel_dim},
        /*combine_degree=*/degree,
    };

    SubParallelComputationGraph original_pcg = [&] {
      ParallelComputationGraph pcg = empty_parallel_computation_graph();

      ParallelLayerAddedResult input_added =
          pcg_add_input_layer(pcg, input_shape);

      parallel_tensor_guid_t t_input = get_only(input_added.outputs);

      ParallelLayerAddedResult relu_added = add_parallel_layer(
          pcg, make_layer_attrs(relu_attrs, relu_match), {t_input}, {});

      return sub_pcg_from_full_pcg(pcg);
    }();

    PCGPatternMatch match = [&] {
      parallel_layer_guid_t match_layer =
          get_parallel_layer_by_name(original_pcg, relu_match);
      open_parallel_tensor_guid_t match_layer_input =
          get_layer_inputs(original_pcg, match_layer).at(0);

      return PCGPatternMatch{
          bidict<PatternNode, parallel_layer_guid_t>{
              {PatternNode{Node{0}}, match_layer},
          },
          std::unordered_map<PatternInput, open_parallel_tensor_guid_t>{{
              PatternInput{DataflowGraphInput{0}},
              match_layer_input,
          }},
      };
    }();

    SubParallelComputationGraph result =
        apply_substitution(original_pcg, sub, match);

    SubParallelComputationGraph correct = [&] {
      ParallelComputationGraph pcg = empty_parallel_computation_graph();

      ParallelLayerAddedResult input_added =
          pcg_add_input_layer(pcg, input_shape);

      parallel_tensor_guid_t t_input = get_only(input_added.outputs);

      ParallelLayerAddedResult partition_input_added = add_parallel_layer(
          pcg, make_layer_attrs(partition_input_attrs), {t_input}, {});
      parallel_tensor_guid_t t_partitioned_input =
          get_only(partition_input_added.outputs);

      ParallelLayerAddedResult partition_relu_added = add_parallel_layer(
          pcg, make_layer_attrs(relu_attrs), {t_partitioned_input}, {});
      parallel_tensor_guid_t t_partitioned_relu =
          get_only(partition_relu_added.outputs);

      ParallelLayerAddedResult combine_added = add_parallel_layer(
          pcg, make_layer_attrs(combine_attrs), {t_partitioned_relu}, {});
      parallel_tensor_guid_t t_combine = get_only(combine_added.outputs);

      return sub_pcg_from_full_pcg(pcg);
    }();

    CHECK(sub_pcgs_are_isomorphic(result, correct));
  }

  TEST_CASE("create_fuse_linear_activation") {
    Substitution sub = create_fuse_linear_activation(Activation::SIGMOID);
    nonnegative_int in_channels = 24_n;
    nonnegative_int batch_size = 4_n;
    nonnegative_int batch_degree = 2_n;
    std::string mm_match = "mm_match";
    std::string relu_match = "relu_match";

    TensorShape input_shape = TensorShape{
        TensorDims{
            FFOrdered<nonnegative_int>{
                4_n,
                10_n,
            },
        },
        DataType::FLOAT,
    };

    SubParallelComputationGraph pcg = [&] {
      ParallelComputationGraphBuilder b;
      parallel_tensor_guid_t t = b.create_input_tensor(input_shape);
      t = b.dense(t,
                  /*outDim=*/4_n,
                  /*activation=*/std::nullopt,
                  /*use_bias=*/false,
                  /*data_type=*/DataType::FLOAT,
                  /*kernel_initializer=*/std::nullopt,
                  /*bias_initializer=*/std::nullopt,
                  /*name=*/mm_match);
      t = b.relu(t,
                 /*name=*/relu_match);

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
              {PatternNode{Node{0}}, mm_match_layer},
              {PatternNode{Node{1}}, relu_match_layer},
          },
          std::unordered_map<PatternInput, open_parallel_tensor_guid_t>{
              {
                  PatternInput{DataflowGraphInput{0}},
                  mm_match_layer_input_activations,
              },
              {
                  PatternInput{DataflowGraphInput{2}},
                  mm_match_layer_input_weights,
              }},
      };
    }();

    SubParallelComputationGraph result = apply_substitution(pcg, sub, match);

    SubParallelComputationGraph correct = [&] {
      ParallelComputationGraphBuilder b;
      parallel_tensor_guid_t t = b.create_input_tensor(input_shape);
      t = b.dense(t,
                  /*outDim=*/4_n,
                  /*activation=*/Activation::SIGMOID,
                  /*use_bias=*/false,
                  /*data_type=*/DataType::FLOAT,
                  /*kernel_initializer=*/std::nullopt,
                  /*bias_initializer=*/std::nullopt,
                  /*name=*/std::nullopt);

      return sub_pcg_from_full_pcg(b.pcg);
    }();

    CHECK(sub_pcgs_are_isomorphic(result, correct));
  }
}
