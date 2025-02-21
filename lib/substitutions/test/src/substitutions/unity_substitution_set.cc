#include "substitutions/unity_substitution_set.h"
#include "op-attrs/computation_graph_op_attrs.h"
#include "op-attrs/operator_type.h"
#include "op-attrs/ops/combine.h"
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

    CHECK(result.size() == 68);
  }

  TEST_CASE("create_partition_linear_combine") {
    nonnegative_int in_channels = 24_n;
    nonnegative_int batch_size = 4_n;
    nonnegative_int batch_degree = 2_n;
    nonnegative_int num_dims = 2_n;
    nonnegative_int degree = 1_n;
    std::string mm_match = "mm_match";

    SUBCASE("use_bias = false") {
      Substitution sub =
          create_partition_linear_combine(num_dims, degree, false);

      SubParallelComputationGraph original_pcg = [&] {
        ParallelComputationGraph pcg = empty_parallel_computation_graph();

        ParallelTensorShape input_shape = ParallelTensorShape{
            ParallelTensorDims{
                FFOrdered<ShardParallelDim>{
                    ShardParallelDim{batch_size, batch_degree},
                    ShardParallelDim{in_channels, 1_n},
                },
                ReplicaParallelDimSet{
                    SumDegree{1_n},
                    DiscardCopyDegree{1_n},
                },
            },
            DataType::FLOAT,
        };

        ParallelLayerAddedResult input_added = [&] {
          ParallelLayerAttrs input_attrs = ParallelLayerAttrs{
              PCGOperatorAttrs{InputAttrs{}},
              std::nullopt,
          };
          ParallelTensorAttrs input_tensor_attrs =
              ParallelTensorAttrs{input_shape,
                                  /*sync_type=*/std::nullopt,
                                  /*initializer=*/std::nullopt,
                                  CreateGrad::YES};

          return add_parallel_layer(pcg, input_attrs, {}, {input_tensor_attrs});
        }();

        parallel_tensor_guid_t input = get_only(input_added.outputs);

        LinearAttrs linear_op_attrs = LinearAttrs{
            /*out_channels=*/12_n,
            /*use_bias=*/false,
            /*data_type=*/DataType::FLOAT,
            /*activation=*/std::nullopt,
            /*regularizer=*/std::nullopt,
        };

        ParallelTensorShape linear_input_shape =
            get_parallel_tensor_attrs(pcg, input).shape;

        ParallelLayerAddedResult linear_weight_added = [&] {
          ParallelTensorShape projection_weight_shape = throw_if_unexpected(
              get_projection_shape(linear_op_attrs, linear_input_shape));

          TensorShape unpar_projection_shape =
              get_reduced_shape(projection_weight_shape);
          ParallelTensorShape raw_projection_weight_shape =
              lift_to_parallel(unpar_projection_shape);

          ParallelLayerAttrs raw_projection_weight_attrs = ParallelLayerAttrs{
              PCGOperatorAttrs{WeightAttrs{unpar_projection_shape}},
              std::nullopt,
          };
          ParallelTensorAttrs raw_projection_tensor_attrs =
              ParallelTensorAttrs{raw_projection_weight_shape,
                                  /*sync_type=*/std::nullopt,
                                  /*initializer=*/std::nullopt,
                                  CreateGrad::YES};

          return add_parallel_layer(pcg,
                                    raw_projection_weight_attrs,
                                    {},
                                    {raw_projection_tensor_attrs});
        }();

        parallel_tensor_guid_t weight = get_only(linear_weight_added.outputs);

        ParallelTensorShape output_shape = throw_if_unexpected(
            get_output_shape(linear_op_attrs, linear_input_shape));

        ParallelLayerAttrs layer_attrs = ParallelLayerAttrs{
            PCGOperatorAttrs{linear_op_attrs},
            mm_match,
        };
        ParallelTensorAttrs output_tensor_attrs =
            ParallelTensorAttrs{output_shape,
                                /*sync_type=*/std::nullopt,
                                /*initializer=*/std::nullopt,
                                CreateGrad::YES};

        add_parallel_layer(
            pcg, layer_attrs, {input, weight}, {output_tensor_attrs});

        return sub_pcg_from_full_pcg(pcg);
      }();

      PCGPatternMatch match = [&] {
        parallel_layer_guid_t mm_match_layer =
            get_parallel_layer_by_name(original_pcg, mm_match);
        open_parallel_tensor_guid_t mm_match_layer_input_activations =
            get_layer_inputs(original_pcg, mm_match_layer).at(0);
        open_parallel_tensor_guid_t mm_match_layer_input_weights =
            get_layer_inputs(original_pcg, mm_match_layer).at(1);

        return PCGPatternMatch{
            bidict<PatternNode, parallel_layer_guid_t>{
                {PatternNode{Node{0}}, mm_match_layer},
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

      SubParallelComputationGraph result =
          apply_substitution(original_pcg, sub, match);

      SubParallelComputationGraph correct = [&] {
        ParallelComputationGraph pcg = empty_parallel_computation_graph();

        ParallelTensorShape input_shape = ParallelTensorShape{
            ParallelTensorDims{
                FFOrdered<ShardParallelDim>{
                    ShardParallelDim{batch_size, batch_degree},
                    ShardParallelDim{in_channels, 1_n},
                },
                ReplicaParallelDimSet{
                    SumDegree{1_n},
                    DiscardCopyDegree{1_n},
                },
            },
            DataType::FLOAT,
        };

        ParallelLayerAddedResult input_added = [&] {
          ParallelLayerAttrs input_attrs = ParallelLayerAttrs{
              PCGOperatorAttrs{InputAttrs{}},
              std::nullopt,
          };
          ParallelTensorAttrs input_tensor_attrs =
              ParallelTensorAttrs{input_shape,
                                  /*sync_type=*/std::nullopt,
                                  /*initializer=*/std::nullopt,
                                  CreateGrad::YES};

          return add_parallel_layer(pcg, input_attrs, {}, {input_tensor_attrs});
        }();

        parallel_tensor_guid_t input = get_only(input_added.outputs);

        RepartitionAttrs repartition_attrs =
            RepartitionAttrs{/*repartition_dim=*/ff_dim_t{1_n},
                             /*repartition_degree=*/degree};

        ParallelLayerAddedResult partitioned_input = [&] {
          ParallelTensorShape output_shape = throw_if_unexpected(
              get_output_shape(repartition_attrs, input_shape));

          ParallelLayerAttrs layer_attrs = ParallelLayerAttrs{
              PCGOperatorAttrs{repartition_attrs},
              std::nullopt,
          };
          ParallelTensorAttrs output_tensor_attrs =
              ParallelTensorAttrs{output_shape,
                                  /*sync_type=*/std::nullopt,
                                  /*initializer=*/std::nullopt,
                                  CreateGrad::YES};

          return add_parallel_layer(
              pcg, layer_attrs, {input}, {output_tensor_attrs});
        }();

        LinearAttrs linear_attrs = LinearAttrs{
            /*out_channels=*/12_n,
            /*use_bias=*/false,
            /*data_type=*/DataType::FLOAT,
            /*activation=*/std::nullopt,
            /*regularizer=*/std::nullopt,
        };

        parallel_tensor_guid_t linear_input =
            get_only(partitioned_input.outputs);

        ParallelTensorShape linear_input_shape =
            get_parallel_tensor_attrs(pcg, linear_input).shape;

        ParallelLayerAddedResult replicated_weight_added = [&] {
          ParallelTensorShape projection_weight_shape = throw_if_unexpected(
              get_projection_shape(linear_attrs, linear_input_shape));

          TensorShape unpar_projection_shape =
              get_reduced_shape(projection_weight_shape);
          ParallelTensorShape raw_projection_weight_shape =
              lift_to_parallel(unpar_projection_shape);

          ParallelLayerAttrs raw_projection_weight_attrs = ParallelLayerAttrs{
              PCGOperatorAttrs{WeightAttrs{unpar_projection_shape}},
              std::nullopt,
          };
          ParallelTensorAttrs raw_projection_tensor_attrs =
              ParallelTensorAttrs{raw_projection_weight_shape,
                                  /*sync_type=*/std::nullopt,
                                  /*initializer=*/std::nullopt,
                                  CreateGrad::YES};

          ParallelLayerAddedResult raw_weight_added =
              add_parallel_layer(pcg,
                                 raw_projection_weight_attrs,
                                 {},
                                 {raw_projection_tensor_attrs});

          ReplicateAttrs replicate_attrs = ReplicateAttrs{degree};

          ParallelLayerAttrs replicate_layer_attrs = ParallelLayerAttrs{
              PCGOperatorAttrs{replicate_attrs},
              std::nullopt,
          };

          ParallelTensorAttrs replicated_projection_tensor_attrs =
              ParallelTensorAttrs{get_output_shape(replicate_attrs,
                                                   raw_projection_weight_shape),
                                  /*sync_type=*/std::nullopt,
                                  /*initializer=*/std::nullopt,
                                  CreateGrad::YES};

          return add_parallel_layer(pcg,
                                    replicate_layer_attrs,
                                    {get_only(raw_weight_added.outputs)},
                                    {replicated_projection_tensor_attrs});
        }();
        parallel_tensor_guid_t linear_weight =
            get_only(replicated_weight_added.outputs);

        ParallelLayerAddedResult linear_output = [&] {
          ParallelTensorShape output_shape = throw_if_unexpected(
              get_output_shape(linear_attrs, linear_input_shape));

          ParallelLayerAttrs linear_layer_attrs = ParallelLayerAttrs{
              PCGOperatorAttrs{linear_attrs},
              std::nullopt,
          };
          ParallelTensorAttrs output_tensor_attrs =
              ParallelTensorAttrs{output_shape,
                                  /*sync_type=*/std::nullopt,
                                  /*initializer=*/std::nullopt,
                                  CreateGrad::YES};

          return add_parallel_layer(pcg,
                                    linear_layer_attrs,
                                    {linear_input, linear_weight},
                                    {output_tensor_attrs});
        }();

        CombineAttrs combine_op_attrs = CombineAttrs{
            /*combine_dim=*/ff_dim_t{
                nonnegative_int{num_dims.unwrap_nonnegative() - 1}},
            /*combine_degree=*/degree,
        };

        parallel_tensor_guid_t combine_input = get_only(linear_output.outputs);

        ParallelTensorShape combine_input_shape =
            get_parallel_tensor_attrs(pcg, combine_input).shape;

        ParallelTensorShape output_shape = throw_if_unexpected(
            get_output_shape(combine_op_attrs, combine_input_shape));

        ParallelLayerAttrs combine_layer_attrs = ParallelLayerAttrs{
            PCGOperatorAttrs{combine_op_attrs},
            std::nullopt,
        };
        ParallelTensorAttrs output_tensor_attrs =
            ParallelTensorAttrs{output_shape,
                                /*sync_type=*/std::nullopt,
                                /*initializer=*/std::nullopt,
                                CreateGrad::YES};

        add_parallel_layer(
            pcg, combine_layer_attrs, {combine_input}, {output_tensor_attrs});

        return sub_pcg_from_full_pcg(pcg);
      }();

      CHECK(sub_pcgs_are_isomorphic(result, correct));
    }
  }

  TEST_CASE("create_replicate_linear_combine") {
    nonnegative_int in_channels = 24_n;
    nonnegative_int batch_size = 4_n;
    nonnegative_int batch_degree = 2_n;
    nonnegative_int num_dims = 2_n;
    nonnegative_int replicate_degree = 2_n;
    std::string mm_match = "mm_match";

    SUBCASE("use_bias = false") {
      Substitution sub =
          create_replicate_linear_combine(num_dims, replicate_degree, false);

      SubParallelComputationGraph original_pcg = [&] {
        ParallelComputationGraph pcg = empty_parallel_computation_graph();

        ParallelTensorShape input_shape = ParallelTensorShape{
            ParallelTensorDims{
                FFOrdered<ShardParallelDim>{
                    ShardParallelDim{batch_size, batch_degree},
                    ShardParallelDim{in_channels, 1_n},
                },
                ReplicaParallelDimSet{
                    SumDegree{1_n},
                    DiscardCopyDegree{1_n},
                },
            },
            DataType::FLOAT,
        };

        ParallelLayerAddedResult input_added = [&] {
          ParallelLayerAttrs input_attrs = ParallelLayerAttrs{
              PCGOperatorAttrs{InputAttrs{}},
              std::nullopt,
          };
          ParallelTensorAttrs input_tensor_attrs =
              ParallelTensorAttrs{input_shape,
                                  /*sync_type=*/std::nullopt,
                                  /*initializer=*/std::nullopt,
                                  CreateGrad::YES};

          return add_parallel_layer(pcg, input_attrs, {}, {input_tensor_attrs});
        }();

        parallel_tensor_guid_t input = get_only(input_added.outputs);

        LinearAttrs linear_op_attrs = LinearAttrs{
            /*out_channels=*/12_n,
            /*use_bias=*/false,
            /*data_type=*/DataType::FLOAT,
            /*activation=*/std::nullopt,
            /*regularizer=*/std::nullopt,
        };

        ParallelTensorShape linear_input_shape =
            get_parallel_tensor_attrs(pcg, input).shape;

        ParallelLayerAddedResult linear_weight_added = [&] {
          ParallelTensorShape projection_weight_shape = throw_if_unexpected(
              get_projection_shape(linear_op_attrs, linear_input_shape));

          TensorShape unpar_projection_shape =
              get_reduced_shape(projection_weight_shape);
          ParallelTensorShape raw_projection_weight_shape =
              lift_to_parallel(unpar_projection_shape);

          ParallelLayerAttrs raw_projection_weight_attrs = ParallelLayerAttrs{
              PCGOperatorAttrs{WeightAttrs{unpar_projection_shape}},
              std::nullopt,
          };
          ParallelTensorAttrs raw_projection_tensor_attrs =
              ParallelTensorAttrs{raw_projection_weight_shape,
                                  /*sync_type=*/std::nullopt,
                                  /*initializer=*/std::nullopt,
                                  CreateGrad::YES};

          return add_parallel_layer(pcg,
                                    raw_projection_weight_attrs,
                                    {},
                                    {raw_projection_tensor_attrs});
        }();

        parallel_tensor_guid_t weight = get_only(linear_weight_added.outputs);

        ParallelTensorShape output_shape = throw_if_unexpected(
            get_output_shape(linear_op_attrs, linear_input_shape));

        ParallelLayerAttrs layer_attrs = ParallelLayerAttrs{
            PCGOperatorAttrs{linear_op_attrs},
            mm_match,
        };
        ParallelTensorAttrs output_tensor_attrs =
            ParallelTensorAttrs{output_shape,
                                /*sync_type=*/std::nullopt,
                                /*initializer=*/std::nullopt,
                                CreateGrad::YES};

        add_parallel_layer(
            pcg, layer_attrs, {input, weight}, {output_tensor_attrs});

        return sub_pcg_from_full_pcg(pcg);
      }();

      PCGPatternMatch match = [&] {
        parallel_layer_guid_t mm_match_layer =
            get_parallel_layer_by_name(original_pcg, mm_match);
        open_parallel_tensor_guid_t mm_match_layer_input_activations =
            get_layer_inputs(original_pcg, mm_match_layer).at(0);
        open_parallel_tensor_guid_t mm_match_layer_input_weights =
            get_layer_inputs(original_pcg, mm_match_layer).at(1);

        return PCGPatternMatch{
            bidict<PatternNode, parallel_layer_guid_t>{
                {PatternNode{Node{0}}, mm_match_layer},
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

      SubParallelComputationGraph result =
          apply_substitution(original_pcg, sub, match);

      SubParallelComputationGraph correct = [&] {
        ParallelComputationGraph pcg = empty_parallel_computation_graph();

        ParallelTensorShape input_shape = ParallelTensorShape{
            ParallelTensorDims{
                FFOrdered<ShardParallelDim>{
                    ShardParallelDim{batch_size, batch_degree},
                    ShardParallelDim{in_channels, 1_n},
                },
                ReplicaParallelDimSet{
                    SumDegree{1_n},
                    DiscardCopyDegree{1_n},
                },
            },
            DataType::FLOAT,
        };

        ParallelLayerAddedResult input_added = [&] {
          ParallelLayerAttrs input_attrs = ParallelLayerAttrs{
              PCGOperatorAttrs{InputAttrs{}},
              std::nullopt,
          };
          ParallelTensorAttrs input_tensor_attrs =
              ParallelTensorAttrs{input_shape,
                                  /*sync_type=*/std::nullopt,
                                  /*initializer=*/std::nullopt,
                                  CreateGrad::YES};

          return add_parallel_layer(pcg, input_attrs, {}, {input_tensor_attrs});
        }();

        parallel_tensor_guid_t input = get_only(input_added.outputs);

        ReplicateAttrs replicate_attrs = ReplicateAttrs{replicate_degree};

        ParallelLayerAddedResult replicated_input = [&] {
          ParallelTensorShape output_shape =
              get_output_shape(replicate_attrs, input_shape);

          ParallelLayerAttrs layer_attrs = ParallelLayerAttrs{
              PCGOperatorAttrs{replicate_attrs},
              std::nullopt,
          };
          ParallelTensorAttrs output_tensor_attrs =
              ParallelTensorAttrs{output_shape,
                                  /*sync_type=*/std::nullopt,
                                  /*initializer=*/std::nullopt,
                                  CreateGrad::YES};

          return add_parallel_layer(
              pcg, layer_attrs, {input}, {output_tensor_attrs});
        }();

        LinearAttrs linear_attrs = LinearAttrs{
            /*out_channels=*/12_n,
            /*use_bias=*/false,
            /*data_type=*/DataType::FLOAT,
            /*activation=*/std::nullopt,
            /*regularizer=*/std::nullopt,
        };

        parallel_tensor_guid_t linear_input =
            get_only(replicated_input.outputs);

        ParallelTensorShape linear_input_shape =
            get_parallel_tensor_attrs(pcg, linear_input).shape;

        ParallelLayerAddedResult partitioned_weight_added = [&] {
          ParallelTensorShape projection_weight_shape = throw_if_unexpected(
              get_projection_shape(linear_attrs, linear_input_shape));

          TensorShape unpar_projection_shape =
              get_reduced_shape(projection_weight_shape);
          ParallelTensorShape raw_projection_weight_shape =
              lift_to_parallel(unpar_projection_shape);

          ParallelLayerAttrs raw_projection_weight_attrs = ParallelLayerAttrs{
              PCGOperatorAttrs{WeightAttrs{unpar_projection_shape}},
              std::nullopt,
          };
          ParallelTensorAttrs raw_projection_tensor_attrs =
              ParallelTensorAttrs{raw_projection_weight_shape,
                                  /*sync_type=*/std::nullopt,
                                  /*initializer=*/std::nullopt,
                                  CreateGrad::YES};

          ParallelLayerAddedResult raw_weight_added =
              add_parallel_layer(pcg,
                                 raw_projection_weight_attrs,
                                 {},
                                 {raw_projection_tensor_attrs});

          RepartitionAttrs repartition_attrs =
              RepartitionAttrs{/*repartition_dim=*/ff_dim_t{1_n},
                               /*repartition_degree=*/replicate_degree};

          ParallelLayerAttrs repartition_layer_attrs = ParallelLayerAttrs{
              PCGOperatorAttrs{repartition_attrs},
              std::nullopt,
          };

          ParallelTensorAttrs partitioned_projection_tensor_attrs =
              ParallelTensorAttrs{
                  throw_if_unexpected(get_output_shape(
                      repartition_attrs, raw_projection_weight_shape)),
                  /*sync_type=*/std::nullopt,
                  /*initializer=*/std::nullopt,
                  CreateGrad::YES};

          return add_parallel_layer(pcg,
                                    repartition_layer_attrs,
                                    {get_only(raw_weight_added.outputs)},
                                    {partitioned_projection_tensor_attrs});
        }();
        parallel_tensor_guid_t linear_weight =
            get_only(partitioned_weight_added.outputs);

        ParallelLayerAddedResult linear_output = [&] {
          ParallelTensorShape output_shape = throw_if_unexpected(
              get_output_shape(linear_attrs, linear_input_shape));

          ParallelLayerAttrs linear_layer_attrs = ParallelLayerAttrs{
              PCGOperatorAttrs{linear_attrs},
              std::nullopt,
          };
          ParallelTensorAttrs output_tensor_attrs =
              ParallelTensorAttrs{output_shape,
                                  /*sync_type=*/std::nullopt,
                                  /*initializer=*/std::nullopt,
                                  CreateGrad::YES};

          return add_parallel_layer(pcg,
                                    linear_layer_attrs,
                                    {linear_input, linear_weight},
                                    {output_tensor_attrs});
        }();

        CombineAttrs combine_op_attrs = CombineAttrs{
            /*combine_dim=*/ff_dim_t{
                nonnegative_int{num_dims.unwrap_nonnegative() - 1}},
            /*combine_degree=*/replicate_degree,
        };

        parallel_tensor_guid_t combine_input = get_only(linear_output.outputs);

        ParallelTensorShape combine_input_shape =
            get_parallel_tensor_attrs(pcg, combine_input).shape;

        ParallelTensorShape output_shape = throw_if_unexpected(
            get_output_shape(combine_op_attrs, combine_input_shape));

        ParallelLayerAttrs combine_layer_attrs = ParallelLayerAttrs{
            PCGOperatorAttrs{combine_op_attrs},
            std::nullopt,
        };
        ParallelTensorAttrs output_tensor_attrs =
            ParallelTensorAttrs{output_shape,
                                /*sync_type=*/std::nullopt,
                                /*initializer=*/std::nullopt,
                                CreateGrad::YES};

        add_parallel_layer(
            pcg, combine_layer_attrs, {combine_input}, {output_tensor_attrs});

        return sub_pcg_from_full_pcg(pcg);
      }();

      CHECK(sub_pcgs_are_isomorphic(result, correct));
    }
  }

  TEST_CASE("create_partition_attention_combine") {
    nonnegative_int embed_dim = 8_n;
    nonnegative_int num_heads = 6_n;
    nonnegative_int degree = 1_n;
    std::string mm_match = "MULTIHEAD_ATTENTION";

    Substitution sub = create_partition_attention_combine(num_heads, degree);

    ShardParallelDim batch_dim = ShardParallelDim{12_n, 2_n};
    ShardParallelDim sequence_dim = ShardParallelDim{16_n, 1_n};
    ShardParallelDim feature_dim = ShardParallelDim{10_n, 1_n};
    ParallelTensorShape query_shape = ParallelTensorShape{
        ParallelTensorDims{
            FFOrdered<ShardParallelDim>{
                batch_dim,
                sequence_dim,
                feature_dim,
            },
            ReplicaParallelDimSet{
                SumDegree{1_n},
                DiscardCopyDegree{1_n},
            },
        },
        DataType::FLOAT,
    };

    SubParallelComputationGraph pcg = [&] {
      ParallelComputationGraphBuilder b;
      parallel_tensor_guid_t input = b.create_input_tensor(query_shape);
      parallel_tensor_guid_t output =
          b.multihead_attention(input, input, input, embed_dim, num_heads);
      return sub_pcg_from_full_pcg(b.pcg);
    }();

    PCGPatternMatch match = [&] {
      parallel_layer_guid_t mm_match_layer =
          get_parallel_layer_by_name(pcg, mm_match);
      open_parallel_tensor_guid_t mm_match_layer_input_activations =
          get_layer_inputs(pcg, mm_match_layer).at(0);
      open_parallel_tensor_guid_t mm_match_layer_input_query_weights =
          get_layer_inputs(pcg, mm_match_layer).at(3);
      open_parallel_tensor_guid_t mm_match_layer_input_key_weights =
          get_layer_inputs(pcg, mm_match_layer).at(4);
      open_parallel_tensor_guid_t mm_match_layer_input_value_weights =
          get_layer_inputs(pcg, mm_match_layer).at(5);

      return PCGPatternMatch{
          bidict<PatternNode, parallel_layer_guid_t>{
              {PatternNode{Node{0}}, mm_match_layer},
          },
          std::unordered_map<PatternInput, open_parallel_tensor_guid_t>{
              {
                  PatternInput{DataflowGraphInput{0}},
                  mm_match_layer_input_activations,
              },
              {
                  PatternInput{DataflowGraphInput{2}},
                  mm_match_layer_input_query_weights,
              },
              {
                  PatternInput{DataflowGraphInput{4}},
                  mm_match_layer_input_key_weights,
              },
              {
                  PatternInput{DataflowGraphInput{6}},
                  mm_match_layer_input_value_weights,
              }},
      };
    }();

    SubParallelComputationGraph correct = [&] {
      ParallelComputationGraphBuilder b;

      parallel_tensor_guid_t t = b.create_input_tensor(query_shape);
      t = b.parallel_partition(t,
                               /*repartition_dim=*/ff_dim_t{1_n},
                               /*repartition_degree=*/degree);
      t = b.multihead_attention(t, t, t, embed_dim, num_heads);
      t = b.parallel_combine(t,
                             /*combine_dim=*/ff_dim_t{2_n},
                             /*combine_degree=*/degree);

      return sub_pcg_from_full_pcg(b.pcg);
    }();

    SubParallelComputationGraph result = apply_substitution(pcg, sub, match);

    CHECK(sub_pcgs_are_isomorphic(result, correct));
  }

  TEST_CASE("create_replicate_attention_reduce") {
    nonnegative_int embed_dim = 8_n;
    nonnegative_int num_heads = 6_n;
    nonnegative_int degree = 1_n;
    std::string mm_match = "MULTIHEAD_ATTENTION";

    Substitution sub = create_replicate_attention_reduce(num_heads, degree);

    ShardParallelDim batch_dim = ShardParallelDim{12_n, 2_n};
    ShardParallelDim sequence_dim = ShardParallelDim{16_n, 1_n};
    ShardParallelDim feature_dim = ShardParallelDim{10_n, 1_n};
    ParallelTensorShape query_shape = ParallelTensorShape{
        ParallelTensorDims{
            FFOrdered<ShardParallelDim>{
                batch_dim,
                sequence_dim,
                feature_dim,
            },
            ReplicaParallelDimSet{
                SumDegree{1_n},
                DiscardCopyDegree{1_n},
            },
        },
        DataType::FLOAT,
    };

    SubParallelComputationGraph pcg = [&] {
      ParallelComputationGraphBuilder b;
      parallel_tensor_guid_t input = b.create_input_tensor(query_shape);
      parallel_tensor_guid_t output =
          b.multihead_attention(input, input, input, embed_dim, num_heads);
      return sub_pcg_from_full_pcg(b.pcg);
    }();

    PCGPatternMatch match = [&] {
      parallel_layer_guid_t mm_match_layer =
          get_parallel_layer_by_name(pcg, mm_match);
      open_parallel_tensor_guid_t mm_match_layer_input_activations =
          get_layer_inputs(pcg, mm_match_layer).at(0);
      open_parallel_tensor_guid_t mm_match_layer_input_query_weights =
          get_layer_inputs(pcg, mm_match_layer).at(3);
      open_parallel_tensor_guid_t mm_match_layer_input_key_weights =
          get_layer_inputs(pcg, mm_match_layer).at(4);
      open_parallel_tensor_guid_t mm_match_layer_input_value_weights =
          get_layer_inputs(pcg, mm_match_layer).at(5);

      return PCGPatternMatch{
          bidict<PatternNode, parallel_layer_guid_t>{
              {PatternNode{Node{0}}, mm_match_layer},
          },
          std::unordered_map<PatternInput, open_parallel_tensor_guid_t>{
              {
                  PatternInput{DataflowGraphInput{0}},
                  mm_match_layer_input_activations,
              },
              {
                  PatternInput{DataflowGraphInput{2}},
                  mm_match_layer_input_query_weights,
              },
              {
                  PatternInput{DataflowGraphInput{4}},
                  mm_match_layer_input_key_weights,
              },
              {
                  PatternInput{DataflowGraphInput{6}},
                  mm_match_layer_input_value_weights,
              }},
      };
    }();

    SubParallelComputationGraph correct = [&] {
      ParallelComputationGraphBuilder b;

      parallel_tensor_guid_t t = b.create_input_tensor(query_shape);
      t = b.parallel_replicate(t, degree);
      t = b.multihead_attention(t, t, t, embed_dim, num_heads);
      t = b.parallel_reduce(t, degree);

      return sub_pcg_from_full_pcg(b.pcg);
    }();

    SubParallelComputationGraph result = apply_substitution(pcg, sub, match);

    CHECK(sub_pcgs_are_isomorphic(result, correct));
  }

  TEST_CASE("create_partition_conv2d_combine") {
    nonnegative_int batch_size = 2_n;
    nonnegative_int batch_degree = 2_n;
    nonnegative_int num_dims = 4_n;
    nonnegative_int degree = 1_n;
    std::string mm_match = "mm_match";

    Substitution sub = create_partition_conv2d_combine(num_dims, degree);

    ParallelTensorShape input_shape = ParallelTensorShape{
        ParallelTensorDims{
            FFOrdered<ShardParallelDim>{
                ShardParallelDim{batch_size, batch_degree},
                ShardParallelDim{3_n, 1_n},
                ShardParallelDim{10_n, 1_n},
                ShardParallelDim{10_n, 1_n},
            },
            ReplicaParallelDimSet{
                SumDegree{1_n},
                DiscardCopyDegree{1_n},
            },
        },
        DataType::FLOAT,
    };

    SubParallelComputationGraph pcg = [&] {
      ParallelComputationGraphBuilder b;
      parallel_tensor_guid_t t = b.create_input_tensor(input_shape);
      nonnegative_int outChannels = 6_n;
      nonnegative_int kernelH = 5_n;
      nonnegative_int kernelW = 4_n;
      nonnegative_int strideH = 3_n;
      nonnegative_int strideW = 2_n;
      nonnegative_int paddingH = 1_n;
      nonnegative_int paddingW = 0_n;
      t = b.conv2d(t,
                   /*outChannels=*/outChannels,
                   /*kernelH=*/kernelH,
                   /*kernelW=*/kernelW,
                   /*strideH=*/strideH,
                   /*strideW=*/strideW,
                   /*paddingH=*/paddingH,
                   /*paddingW=*/paddingW,
                   std::nullopt,
                   1_n,
                   false,
                   std::nullopt,
                   std::nullopt,
                   std::nullopt,
                   mm_match);
      return sub_pcg_from_full_pcg(b.pcg);
    }();

    PCGPatternMatch match = [&] {
      parallel_layer_guid_t mm_match_layer =
          get_parallel_layer_by_name(pcg, mm_match);
      open_parallel_tensor_guid_t mm_match_layer_input_activations =
          get_layer_inputs(pcg, mm_match_layer).at(0);
      open_parallel_tensor_guid_t mm_match_layer_input_weights =
          get_layer_inputs(pcg, mm_match_layer).at(1);

      return PCGPatternMatch{
          bidict<PatternNode, parallel_layer_guid_t>{
              {PatternNode{Node{0}}, mm_match_layer},
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
      nonnegative_int outChannels = 6_n;
      nonnegative_int kernelH = 5_n;
      nonnegative_int kernelW = 4_n;
      nonnegative_int strideH = 3_n;
      nonnegative_int strideW = 2_n;
      nonnegative_int paddingH = 1_n;
      nonnegative_int paddingW = 0_n;
      t = b.parallel_partition(t,
                               /*repartition_dim=*/ff_dim_t{1_n},
                               /*repartition_degree=*/degree);
      t = b.conv2d(t,
                   /*outChannels=*/outChannels,
                   /*kernelH=*/kernelH,
                   /*kernelW=*/kernelW,
                   /*strideH=*/strideH,
                   /*strideW=*/strideW,
                   /*paddingH=*/paddingH,
                   /*paddingW=*/paddingW,
                   std::nullopt,
                   1_n,
                   false,
                   std::nullopt,
                   std::nullopt,
                   std::nullopt,
                   mm_match);
      t = b.parallel_combine(
          t,
          /*combine_dim=*/
          ff_dim_t{
              nonnegative_int{num_dims.unwrap_nonnegative() - 1},
          },
          /*combine_degree=*/degree);
      return sub_pcg_from_full_pcg(b.pcg);
    }();

    CHECK(sub_pcgs_are_isomorphic(result, correct));
  }

  TEST_CASE("create_partition_relu_combine") {
    nonnegative_int degree = 1_n;
    ff_dim_t parallel_dim = ff_dim_t{1_n};
    std::string relu_match = "relu_match";

    Substitution sub = create_partition_relu_combine(parallel_dim, degree);

    ShardParallelDim batch_dim = ShardParallelDim{18_n, 3_n};
    ShardParallelDim feature_dim = ShardParallelDim{32_n, 1_n};

    ParallelTensorShape input_shape = ParallelTensorShape{
        ParallelTensorDims{
            FFOrdered<ShardParallelDim>{
                batch_dim,
                feature_dim,
            },
            ReplicaParallelDimSet{
                SumDegree{1_n},
                DiscardCopyDegree{1_n},
            },
        },
        DataType::FLOAT,
    };

    SubParallelComputationGraph pcg = [&] {
      ParallelComputationGraphBuilder b;
      parallel_tensor_guid_t t = b.create_input_tensor(input_shape);
      t = b.relu(t, relu_match);
      return sub_pcg_from_full_pcg(b.pcg);
    }();

    PCGPatternMatch match = [&] {
      parallel_layer_guid_t relu_match_layer =
          get_parallel_layer_by_name(pcg, relu_match);
      open_parallel_tensor_guid_t relu_match_layer_input =
          get_layer_inputs(pcg, relu_match_layer).at(0);

      return PCGPatternMatch{
          bidict<PatternNode, parallel_layer_guid_t>{
              {PatternNode{Node{0}}, relu_match_layer},
          },
          std::unordered_map<PatternInput, open_parallel_tensor_guid_t>{{
              PatternInput{DataflowGraphInput{0}},
              relu_match_layer_input,
          }},
      };
    }();

    SubParallelComputationGraph result = apply_substitution(pcg, sub, match);

    SubParallelComputationGraph correct = [&] {
      ParallelComputationGraphBuilder b;
      parallel_tensor_guid_t t = b.create_input_tensor(input_shape);
      t = b.parallel_partition(t, parallel_dim, degree);
      t = b.relu(t, relu_match);
      t = b.parallel_combine(t, parallel_dim, degree);
      return sub_pcg_from_full_pcg(b.pcg);
    }();

    CHECK(sub_pcgs_are_isomorphic(result, correct));
  }

  TEST_CASE("create_partition_add_combine") {
    nonnegative_int degree = 1_n;
    ff_dim_t parallel_dim = ff_dim_t{1_n};
    std::string add_match = "add_match";

    Substitution sub = create_partition_add_combine(parallel_dim, degree);

    ShardParallelDim d1 = ShardParallelDim{10_n, 2_n};
    ShardParallelDim d2 = ShardParallelDim{15_n, 3_n};

    ParallelTensorShape lhs_shape = ParallelTensorShape{
        ParallelTensorDims{
            FFOrdered<ShardParallelDim>{
                ShardParallelDim{10_n, 2_n},
                ShardParallelDim{15_n, 3_n},
            },
            ReplicaParallelDimSet{
                SumDegree{2_n},
                DiscardCopyDegree{1_n},
            },
        },
        DataType::FLOAT,
    };

    ParallelTensorShape rhs_shape = lhs_shape;

    SubParallelComputationGraph pcg = [&] {
      ParallelComputationGraphBuilder b;
      parallel_tensor_guid_t lhs = b.create_input_tensor(lhs_shape);
      parallel_tensor_guid_t rhs = b.create_input_tensor(rhs_shape);
      parallel_tensor_guid_t out = b.add(lhs, rhs, add_match);
      return sub_pcg_from_full_pcg(b.pcg);
    }();

    PCGPatternMatch match = [&] {
      parallel_layer_guid_t add_match_layer =
          get_parallel_layer_by_name(pcg, add_match);
      open_parallel_tensor_guid_t add_match_layer_input1 =
          get_layer_inputs(pcg, add_match_layer).at(0);
      open_parallel_tensor_guid_t add_match_layer_input2 =
          get_layer_inputs(pcg, add_match_layer).at(1);

      return PCGPatternMatch{
          bidict<PatternNode, parallel_layer_guid_t>{
              {PatternNode{Node{0}}, add_match_layer},
          },
          std::unordered_map<PatternInput, open_parallel_tensor_guid_t>{
              {
                  PatternInput{DataflowGraphInput{0}},
                  add_match_layer_input1,
              },
              {
                  PatternInput{DataflowGraphInput{2}},
                  add_match_layer_input2,
              }}};
    }();

    SubParallelComputationGraph result = apply_substitution(pcg, sub, match);

    SubParallelComputationGraph correct = [&] {
      ParallelComputationGraphBuilder b;
      parallel_tensor_guid_t lhs = b.create_input_tensor(lhs_shape);
      parallel_tensor_guid_t rhs = b.create_input_tensor(rhs_shape);
      lhs = b.parallel_partition(lhs, parallel_dim, degree);
      rhs = b.parallel_partition(rhs, parallel_dim, degree);
      parallel_tensor_guid_t t = b.add(lhs, rhs);
      t = b.parallel_combine(t, parallel_dim, degree);
      return sub_pcg_from_full_pcg(b.pcg);
    }();

    CHECK(sub_pcgs_are_isomorphic(result, correct));
  }

  // TEST_CASE("create_partition_softmax_combine") {
  //   CHECK(false);
  // }

  TEST_CASE("create_fuse_linear_activation") {
    Substitution sub = create_fuse_linear_activation(Activation::SIGMOID);
    nonnegative_int in_channels = 24_n;
    nonnegative_int batch_size = 4_n;
    nonnegative_int batch_degree = 2_n;
    std::string mm_match = "mm_match";
    std::string relu_match = "relu_match";

    SubParallelComputationGraph pcg = [&] {
      ParallelComputationGraphBuilder b;
      parallel_tensor_guid_t t = b.create_input_tensor(ParallelTensorShape{
          ParallelTensorDims{
              FFOrdered<ShardParallelDim>{
                  ShardParallelDim{batch_size, batch_degree},
                  ShardParallelDim{in_channels, 1_n},
              },
              ReplicaParallelDimSet{
                  SumDegree{1_n},
                  DiscardCopyDegree{1_n},
              },
          },
          DataType::FLOAT,
      });
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
      parallel_tensor_guid_t t = b.create_input_tensor(ParallelTensorShape{
          ParallelTensorDims{
              FFOrdered<ShardParallelDim>{
                  ShardParallelDim{batch_size, batch_degree},
                  ShardParallelDim{in_channels, 1_n},
              },
              ReplicaParallelDimSet{
                  SumDegree{1_n},
                  DiscardCopyDegree{1_n},
              },
          },
          DataType::FLOAT,
      });
      t = b.dense(t,
                  /*outDim=*/16_n,
                  /*activation=*/std::nullopt);
      t = b.gelu(t);
      t = b.dense(t,
                  /*outDim=*/12_n,
                  /*activation=*/Activation::SIGMOID,
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

    CHECK(sub_pcgs_are_isomorphic(result, correct));
  }
}
