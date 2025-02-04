#include "pcg/parallel_computation_graph/parallel_computation_graph_builder.h"
#include "op-attrs/ops/conv_2d.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "pcg/parallel_computation_graph/parallel_computation_graph.h"
#include "pcg/parallel_computation_graph/parallel_layer_attrs.h"
#include "pcg/parallel_computation_graph/parallel_tensor_guid_t.h"
#include "utils/containers/count.h"
#include "utils/containers/generate_map.h"
#include "utils/containers/get_only.h"
#include "utils/containers/items.h"
#include "utils/containers/transform.h"
#include "utils/containers/values.h"
#include "utils/containers/without_nullopts.h"
#include "utils/hash/pair.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

// Stylistically these tests are not great (they're rather complicated
// and hard to read) and should not be used as a model for other FlexFlow
// tests.
//
// Improving them is being tracked in
// https://github.com/flexflow/FlexFlow/issues/1474
TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("ParallelComputationGraphBuilder::add") {
    ParallelComputationGraphBuilder b;

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

    parallel_tensor_guid_t lhs = b.create_input_tensor(lhs_shape);
    parallel_tensor_guid_t rhs = b.create_input_tensor(rhs_shape);

    parallel_tensor_guid_t out = b.add(lhs, rhs);
    parallel_layer_guid_t layer = get_source_layer(out);

    SUBCASE("incoming") {
      std::vector<parallel_tensor_guid_t> result =
          get_incoming_tensors(b.pcg, layer);
      std::vector<parallel_tensor_guid_t> correct = {lhs, rhs};
      CHECK(result == correct);
    }

    SUBCASE("outputs") {
      std::vector<parallel_tensor_guid_t> result =
          get_layer_outputs(b.pcg, layer);
      std::vector<parallel_tensor_guid_t> correct = {out};
      CHECK(result == correct);
    }

    SUBCASE("op attrs") {
      PCGOperatorAttrs result = get_parallel_layer_attrs(b.pcg, layer).op_attrs;
      PCGOperatorAttrs correct = PCGOperatorAttrs{ElementBinaryAttrs{
          OperatorType::EW_ADD, DataType::FLOAT, false, false}};
      CHECK(result == correct);
    }
  }

  TEST_CASE("ParallelComputationGraphBuilder::batch_matmul") {
    ParallelComputationGraphBuilder b;

    ShardParallelDim batch_dim = ShardParallelDim{4_n, 2_n};

    ParallelTensorShape a_shape = ParallelTensorShape{
        ParallelTensorDims{
            FFOrdered<ShardParallelDim>{
                batch_dim,
                ShardParallelDim{10_n, 1_n},
                ShardParallelDim{15_n, 3_n},
            },
            ReplicaParallelDimSet{
                SumDegree{1_n},
                DiscardCopyDegree{1_n},
            },
        },
        DataType::FLOAT,
    };

    ParallelTensorShape b_shape = ParallelTensorShape{
        ParallelTensorDims{
            FFOrdered<ShardParallelDim>{
                batch_dim,
                ShardParallelDim{15_n, 3_n},
                ShardParallelDim{12_n, 1_n},
            },
            ReplicaParallelDimSet{
                SumDegree{1_n},
                DiscardCopyDegree{1_n},
            },
        },
        DataType::FLOAT,
    };

    parallel_tensor_guid_t a_tensor = b.create_input_tensor(a_shape);
    parallel_tensor_guid_t b_tensor = b.create_input_tensor(b_shape);

    parallel_tensor_guid_t out = b.batch_matmul(a_tensor, b_tensor);
    parallel_layer_guid_t layer = get_source_layer(out);

    SUBCASE("incoming") {
      std::vector<parallel_tensor_guid_t> result =
          get_incoming_tensors(b.pcg, layer);
      std::vector<parallel_tensor_guid_t> correct = {a_tensor, b_tensor};
      CHECK(result == correct);
    }

    SUBCASE("outputs") {
      std::vector<parallel_tensor_guid_t> result =
          get_layer_outputs(b.pcg, layer);
      std::vector<parallel_tensor_guid_t> correct = {out};
      CHECK(result == correct);
    }

    SUBCASE("op attrs") {
      PCGOperatorAttrs result = get_parallel_layer_attrs(b.pcg, layer).op_attrs;
      PCGOperatorAttrs correct =
          PCGOperatorAttrs{BatchMatmulAttrs{std::nullopt, std::nullopt}};
      CHECK(result == correct);
    }
  }

  TEST_CASE("ParallelComputationGraphBuilder::cast") {
    ParallelComputationGraphBuilder b;

    ParallelTensorShape input_shape = ParallelTensorShape{
        ParallelTensorDims{
            FFOrdered<ShardParallelDim>{
                ShardParallelDim{10_n, 2_n},
                ShardParallelDim{12_n, 1_n},
            },
            ReplicaParallelDimSet{
                SumDegree{3_n},
                DiscardCopyDegree{1_n},
            },
        },
        DataType::FLOAT,
    };

    DataType output_datatype = DataType::DOUBLE;
    parallel_tensor_guid_t input = b.create_input_tensor(input_shape);
    parallel_tensor_guid_t output = b.cast(input, output_datatype);
    parallel_layer_guid_t layer = get_source_layer(output);

    SUBCASE("incoming") {
      std::vector<parallel_tensor_guid_t> result =
          get_incoming_tensors(b.pcg, layer);
      std::vector<parallel_tensor_guid_t> correct = {input};
      CHECK(result == correct);
    }

    SUBCASE("outputs") {
      std::vector<parallel_tensor_guid_t> result =
          get_layer_outputs(b.pcg, layer);
      std::vector<parallel_tensor_guid_t> correct = {output};
      CHECK(result == correct);

      ParallelTensorShape output_shape =
          get_parallel_tensor_attrs(b.pcg, output).shape;
      CHECK(output_shape.data_type == output_datatype);
    }
  }

  TEST_CASE("ParallelComputationGraphBuilder::conv2d") {
    ParallelComputationGraphBuilder b;

    nonnegative_int batch_size = 2_n;

    TensorShape unpar_input_shape = TensorShape{
        TensorDims{FFOrdered<nonnegative_int>{batch_size, 3_n, 10_n, 10_n}},
        DataType::FLOAT,
    };

    ParallelTensorShape input_shape = lift_to_parallel_with_degrees(
        unpar_input_shape,
        SumDegree{1_n},
        DiscardCopyDegree{1_n},
        FFOrdered<nonnegative_int>{2_n, 1_n, 1_n, 1_n});

    parallel_tensor_guid_t input = b.create_input_tensor(input_shape);

    nonnegative_int outChannels = 6_n;
    nonnegative_int kernelH = 5_n;
    nonnegative_int kernelW = 4_n;
    nonnegative_int strideH = 3_n;
    nonnegative_int strideW = 2_n;
    nonnegative_int paddingH = 1_n;
    nonnegative_int paddingW = 0_n;
    parallel_tensor_guid_t output = b.conv2d(input,
                                             /*outChannels=*/outChannels,
                                             /*kernelH=*/kernelH,
                                             /*kernelW=*/kernelW,
                                             /*strideH=*/strideH,
                                             /*strideW=*/strideW,
                                             /*paddingH=*/paddingH,
                                             /*paddingW=*/paddingW);

    std::unordered_map<parallel_layer_guid_t, ParallelLayerAttrs> layers =
        generate_map(get_parallel_layers(b.pcg),
                     [&](parallel_layer_guid_t const &l) {
                       return get_parallel_layer_attrs(b.pcg, l);
                     });
    CHECK_MESSAGE(layers.size() == 6, "Incorrect layers ", layers);

    auto num_attrs_of_type = [&](OperatorType op_type) -> int {
      return count(values(layers), [&](ParallelLayerAttrs const &l) {
        return get_op_type(l) == op_type;
      });
    };

    int num_weight_attrs = num_attrs_of_type(OperatorType::WEIGHT);
    CHECK(num_weight_attrs == 2);

    int num_input_attrs = num_attrs_of_type(OperatorType::INPUT);
    CHECK(num_input_attrs == 1);

    int num_conv_attrs = num_attrs_of_type(OperatorType::CONV2D);
    CHECK(num_conv_attrs == 1);

    int num_replicate_attrs = num_attrs_of_type(OperatorType::REPLICATE);
    CHECK(num_replicate_attrs == 2);

    parallel_layer_guid_t conv_guid = get_only(without_nullopts(transform(
        vector_of(items(layers)),
        [](std::pair<parallel_layer_guid_t, ParallelLayerAttrs> const &kv)
            -> std::optional<parallel_layer_guid_t> {
          if (get_op_type(kv.second) == OperatorType::CONV2D) {
            return kv.first;
          } else {
            return std::nullopt;
          }
        })));
    Conv2DAttrs conv_attrs = layers.at(conv_guid).op_attrs.get<Conv2DAttrs>();
    Conv2DAttrs correct_attrs = Conv2DAttrs{
        outChannels,
        kernelH,
        kernelW,
        strideH,
        strideW,
        paddingH,
        paddingW,
        /*groups=*/1_n,
        /*activation=*/std::nullopt,
        /*use_bias=*/true,
    };
    CHECK(conv_attrs == correct_attrs);

    ParallelTensorShape correct_output_shape =
        get_output_shape(correct_attrs, input_shape);
    ParallelTensorShape correct_kernel_shape =
        get_kernel_shape(correct_attrs, input_shape);
    ParallelTensorShape correct_bias_shape =
        get_bias_shape(correct_attrs, input_shape);

    std::vector<parallel_tensor_guid_t> conv_incoming =
        get_incoming_tensors(b.pcg, conv_guid);

    parallel_tensor_guid_t conv_input = conv_incoming.at(0);
    ParallelTensorShape conv_input_shape =
        get_parallel_tensor_attrs(b.pcg, conv_input).shape;
    CHECK(conv_input_shape == input_shape);

    parallel_tensor_guid_t conv_kernel = conv_incoming.at(1);
    ParallelTensorShape conv_kernel_shape =
        get_parallel_tensor_attrs(b.pcg, conv_kernel).shape;
    CHECK(conv_kernel_shape == correct_kernel_shape);

    parallel_tensor_guid_t conv_bias = conv_incoming.at(2);
    ParallelTensorShape conv_bias_shape =
        get_parallel_tensor_attrs(b.pcg, conv_bias).shape;
    CHECK(conv_bias_shape == correct_bias_shape);

    std::vector<parallel_tensor_guid_t> conv_outputs =
        get_layer_outputs(b.pcg, conv_guid);
    CHECK(conv_outputs.size() == 1);

    parallel_tensor_guid_t conv_output = get_only(conv_outputs);
    ParallelTensorShape conv_output_shape =
        get_parallel_tensor_attrs(b.pcg, conv_output).shape;
    CHECK(conv_output_shape == correct_output_shape);
  };

  TEST_CASE("ParallelComputationGraphBuilder::dense") {
    ParallelComputationGraphBuilder b;

    ParallelTensorShape input_shape = ParallelTensorShape{
        ParallelTensorDims{
            FFOrdered<ShardParallelDim>{
                ShardParallelDim{10_n, 2_n},
                ShardParallelDim{16_n, 1_n},
            },
            ReplicaParallelDimSet{
                SumDegree{1_n},
                DiscardCopyDegree{1_n},
            },
        },
        DataType::FLOAT,
    };

    nonnegative_int outDim = 14_n;

    parallel_tensor_guid_t input = b.create_input_tensor(input_shape);
    parallel_tensor_guid_t output = b.dense(input,
                                            outDim,
                                            Activation::RELU,
                                            /*use_bias=*/true,
                                            DataType::FLOAT);
    parallel_layer_guid_t layer = get_source_layer(output);

    SUBCASE("incoming") {
      std::vector<parallel_tensor_guid_t> result =
          get_incoming_tensors(b.pcg, layer);
      CHECK(result.at(0) == input);

      CHECK(result.size() == 3);
    }

    SUBCASE("outputs") {
      std::vector<parallel_tensor_guid_t> result =
          get_layer_outputs(b.pcg, layer);
      std::vector<parallel_tensor_guid_t> correct = {output};
      CHECK(result == correct);
    }
  }

  TEST_CASE("ParallelComputationGraphBuilder::embedding") {
    ParallelComputationGraphBuilder b;

    ShardParallelDim batch_dim = ShardParallelDim{12_n, 2_n};
    ShardParallelDim feature_dim = ShardParallelDim{10_n, 1_n};
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
        DataType::INT32,
    };

    parallel_tensor_guid_t input = b.create_input_tensor(input_shape);
    parallel_tensor_guid_t output = b.embedding(input,
                                                /*num_entries=*/32_n,
                                                /*outDim=*/8_n,
                                                AggregateOp::SUM,
                                                DataType::FLOAT);
    parallel_layer_guid_t layer = get_source_layer(output);

    SUBCASE("incoming") {
      std::vector<parallel_tensor_guid_t> result =
          get_incoming_tensors(b.pcg, layer);
      CHECK(result.at(0) == input);

      CHECK(result.size() == 2);
    }

    SUBCASE("outputs") {
      std::vector<parallel_tensor_guid_t> result =
          get_layer_outputs(b.pcg, layer);
      std::vector<parallel_tensor_guid_t> correct = {output};
      CHECK(result == correct);
    }
  }

  TEST_CASE("ParallelComputationGraphBuilder::multihead_attention") {
    ParallelComputationGraphBuilder b;

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

    ParallelTensorShape key_shape = query_shape;
    ParallelTensorShape value_shape = query_shape;

    nonnegative_int embed_dim = 8_n;
    nonnegative_int num_heads = 6_n;

    parallel_tensor_guid_t query = b.create_input_tensor(query_shape);
    parallel_tensor_guid_t key = b.create_input_tensor(key_shape);
    parallel_tensor_guid_t value = b.create_input_tensor(value_shape);
    parallel_tensor_guid_t output =
        b.multihead_attention(query, key, value, embed_dim, num_heads);
    parallel_layer_guid_t layer = get_source_layer(output);

    SUBCASE("incoming") {
      std::vector<parallel_tensor_guid_t> result =
          get_incoming_tensors(b.pcg, layer);
      CHECK(result.at(0) == query);
      CHECK(result.at(1) == key);
      CHECK(result.at(2) == value);
      CHECK(result.size() == 6);
    }

    SUBCASE("outputs") {
      std::vector<parallel_tensor_guid_t> result =
          get_layer_outputs(b.pcg, layer);
      std::vector<parallel_tensor_guid_t> correct = {output};
      CHECK(result == correct);
    }
  }

  TEST_CASE("ParallelComputationGraphBuilder::relu") {
    ParallelComputationGraphBuilder b;

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

    parallel_tensor_guid_t input = b.create_input_tensor(input_shape);
    parallel_tensor_guid_t output = b.relu(input);
    parallel_layer_guid_t layer = get_source_layer(output);

    SUBCASE("incoming") {
      std::vector<parallel_tensor_guid_t> result =
          get_incoming_tensors(b.pcg, layer);
      std::vector<parallel_tensor_guid_t> correct = {input};
      CHECK(result == correct);
    }

    SUBCASE("outputs") {
      std::vector<parallel_tensor_guid_t> result =
          get_layer_outputs(b.pcg, layer);
      std::vector<parallel_tensor_guid_t> correct = {output};
      CHECK(result == correct);
    }
  }

  TEST_CASE("ParallelComputationGraphBuilder::parallel_partition") {
    ParallelComputationGraphBuilder b;

    ShardParallelDim batch_dim = ShardParallelDim{18_n, 2_n};
    ShardParallelDim feature_dim = ShardParallelDim{10_n, 1_n};

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

    parallel_tensor_guid_t input = b.create_input_tensor(input_shape);
    parallel_tensor_guid_t output =
        b.parallel_partition(input, ff_dim_t{nonnegative_int{0}}, 2_n);
    parallel_layer_guid_t layer = get_source_layer(output);

    SUBCASE("incoming") {
      std::vector<parallel_tensor_guid_t> result =
          get_incoming_tensors(b.pcg, layer);
      std::vector<parallel_tensor_guid_t> correct = {input};
      CHECK(result == correct);
    }

    SUBCASE("outputs") {
      std::vector<parallel_tensor_guid_t> result =
          get_layer_outputs(b.pcg, layer);
      std::vector<parallel_tensor_guid_t> correct = {output};
      CHECK(result == correct);
    }
  }

  TEST_CASE("ParallelComputationGraphBuilder::parallel_combine") {
    ParallelComputationGraphBuilder b;

    ShardParallelDim batch_dim = ShardParallelDim{18_n, 2_n};
    ShardParallelDim feature_dim = ShardParallelDim{10_n, 1_n};

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

    parallel_tensor_guid_t input = b.create_input_tensor(input_shape);
    parallel_tensor_guid_t output =
        b.parallel_combine(input, ff_dim_t{nonnegative_int{0}}, 2_n);
    parallel_layer_guid_t layer = get_source_layer(output);

    SUBCASE("incoming") {
      std::vector<parallel_tensor_guid_t> result =
          get_incoming_tensors(b.pcg, layer);
      std::vector<parallel_tensor_guid_t> correct = {input};
      CHECK(result == correct);
    }

    SUBCASE("outputs") {
      std::vector<parallel_tensor_guid_t> result =
          get_layer_outputs(b.pcg, layer);
      std::vector<parallel_tensor_guid_t> correct = {output};
      CHECK(result == correct);
    }
  }

  TEST_CASE("ParallelComputationGraphBuilder::parallel_replicate") {
    ParallelComputationGraphBuilder b;

    ShardParallelDim batch_dim = ShardParallelDim{18_n, 2_n};
    ShardParallelDim feature_dim = ShardParallelDim{10_n, 1_n};

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

    parallel_tensor_guid_t input = b.create_input_tensor(input_shape);
    parallel_tensor_guid_t output = b.parallel_replicate(input, 2_n);
    parallel_layer_guid_t layer = get_source_layer(output);

    SUBCASE("incoming") {
      std::vector<parallel_tensor_guid_t> result =
          get_incoming_tensors(b.pcg, layer);
      std::vector<parallel_tensor_guid_t> correct = {input};
      CHECK(result == correct);
    }

    SUBCASE("outputs") {
      std::vector<parallel_tensor_guid_t> result =
          get_layer_outputs(b.pcg, layer);
      std::vector<parallel_tensor_guid_t> correct = {output};
      CHECK(result == correct);
    }
  }

  TEST_CASE("ParallelComputationGraphBuilder::parallel_reduce") {
    ParallelComputationGraphBuilder b;

    ShardParallelDim batch_dim = ShardParallelDim{18_n, 2_n};
    ShardParallelDim feature_dim = ShardParallelDim{10_n, 1_n};

    ParallelTensorShape input_shape = ParallelTensorShape{
        ParallelTensorDims{
            FFOrdered<ShardParallelDim>{
                batch_dim,
                feature_dim,
            },
            ReplicaParallelDimSet{
                SumDegree{4_n},
                DiscardCopyDegree{1_n},
            },
        },
        DataType::FLOAT,
    };

    parallel_tensor_guid_t input = b.create_input_tensor(input_shape);
    parallel_tensor_guid_t output = b.parallel_reduce(input, 2_n);
    parallel_layer_guid_t layer = get_source_layer(output);

    SUBCASE("incoming") {
      std::vector<parallel_tensor_guid_t> result =
          get_incoming_tensors(b.pcg, layer);
      std::vector<parallel_tensor_guid_t> correct = {input};
      CHECK(result == correct);
    }

    SUBCASE("outputs") {
      std::vector<parallel_tensor_guid_t> result =
          get_layer_outputs(b.pcg, layer);
      std::vector<parallel_tensor_guid_t> correct = {output};
      CHECK(result == correct);
    }
  }
}
