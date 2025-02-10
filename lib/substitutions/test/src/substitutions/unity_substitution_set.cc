#include "substitutions/unity_substitution_set.h"
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
