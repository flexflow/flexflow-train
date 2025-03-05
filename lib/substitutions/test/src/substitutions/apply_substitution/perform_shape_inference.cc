#include "substitutions/apply_substitution/perform_shape_inference.h"
#include "op-attrs/ops/element_unary.h"
#include "op-attrs/ops/linear.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "utils/containers/get_only.h"
#include "utils/graph/instances/unordered_set_labelled_open_dataflow_graph.h"
#include "utils/graph/labelled_open_dataflow_graph/algorithms/get_graph_data.h"
#include "utils/graph/labelled_open_dataflow_graph/labelled_open_dataflow_graph.h"
#include "utils/integer_conversions.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("perform_shape_inference") {
    auto g =
        LabelledOpenDataflowGraph<ParallelLayerAttrs, std::monostate>::create<
            UnorderedSetLabelledOpenDataflowGraph<ParallelLayerAttrs,
                                                  std::monostate>>();

    nonnegative_int in_channels = 24_n;
    nonnegative_int out_channels = 16_n;
    nonnegative_int batch_size = 4_n;
    nonnegative_int batch_degree = 2_n;

    DataflowGraphInput i0 = g.add_input({});
    ParallelTensorShape i0_shape = ParallelTensorShape{
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

    bool use_bias = false;
    LinearAttrs n1_op_attrs = LinearAttrs{
        /*out_channels=*/out_channels,
        /*use_bias=*/use_bias,
        /*data_type=*/DataType::FLOAT,
        /*activation=*/std::nullopt,
        /*regularizer=*/std::nullopt,
    };
    ParallelLayerAttrs n1_attrs = ParallelLayerAttrs{
        /*op_attrs=*/PCGOperatorAttrs{
            n1_op_attrs,
        },
        /*name=*/std::nullopt,
    };

    ElementUnaryAttrs n2_op_attrs = ElementUnaryAttrs{
        /*op_type=*/OperatorType::RELU,
        /*scalar=*/std::nullopt,
    };
    ParallelLayerAttrs n2_attrs = ParallelLayerAttrs{
        /*op_attrs=*/PCGOperatorAttrs{
            n2_op_attrs,
        },
        /*name=*/std::nullopt,
    };

    ParallelTensorShape n1_output_shape =
        throw_if_unexpected(get_output_shape(n1_op_attrs, i0_shape));
    ParallelTensorShape n1_weight_shape =
        throw_if_unexpected(get_projection_shape(n1_op_attrs, i0_shape));
    ParallelTensorShape n2_output_shape =
        throw_if_unexpected(get_output_shape(n2_op_attrs, n1_output_shape));

    ParallelLayerAttrs n1_weight_attrs = ParallelLayerAttrs{
        PCGOperatorAttrs{
            WeightAttrs{
                get_reduced_shape(n1_weight_shape),
                InitializerAttrs{ZeroInitializerAttrs{}},
            },
        },
        std::nullopt,
    };

    ParallelLayerAttrs n1_weight_replicate_attrs = ParallelLayerAttrs{
        PCGOperatorAttrs{
            ReplicateAttrs{batch_degree},
        },
        std::nullopt,
    };

    NodeAddedResult n1_weight_added_result =
        g.add_node(n1_weight_attrs, {}, {{}});
    Node n1_weight_node = n1_weight_added_result.node;
    DataflowOutput n1_weight = get_only(n1_weight_added_result.outputs);

    NodeAddedResult n1_weight_replicate_added_result = g.add_node(
        n1_weight_replicate_attrs, {OpenDataflowValue{n1_weight}}, {{}});
    Node n1_weight_replicate_node = n1_weight_replicate_added_result.node;
    DataflowOutput n1_weight_replicated =
        get_only(n1_weight_replicate_added_result.outputs);

    NodeAddedResult n1_added_result = g.add_node(
        n1_attrs,
        {OpenDataflowValue{i0}, OpenDataflowValue{n1_weight_replicated}},
        {{}});
    Node n1 = n1_added_result.node;
    DataflowOutput o1 = get_only(n1_added_result.outputs);

    NodeAddedResult n2_added_result =
        g.add_node(n2_attrs, {OpenDataflowValue{o1}}, {{}});
    Node n2 = n2_added_result.node;
    DataflowOutput o2 = get_only(n2_added_result.outputs);

    std::unordered_map<DataflowGraphInput, ParallelTensorShape> input_shapes = {
        {i0, i0_shape},
    };

    LabelledOpenDataflowGraphView<ParallelLayerAttrs, ParallelTensorShape>
        result = perform_shape_inference(g, input_shapes);

    LabelledOpenDataflowGraphData<ParallelLayerAttrs, ParallelTensorShape>
        result_data = get_graph_data(result);

    LabelledOpenDataflowGraphData<ParallelLayerAttrs, ParallelTensorShape>
        correct_data = LabelledOpenDataflowGraphData<ParallelLayerAttrs,
                                                     ParallelTensorShape>{
            {
                {n1, n1_attrs},
                {n2, n2_attrs},
                {n1_weight_node, n1_weight_attrs},
                {n1_weight_replicate_node, n1_weight_replicate_attrs},
            },
            {
                OpenDataflowEdge{
                    DataflowInputEdge{
                        i0,
                        DataflowInput{n1, 0_n},
                    },
                },
                OpenDataflowEdge{DataflowEdge{
                    DataflowOutput{n1_weight_node, 0_n},
                    DataflowInput{n1_weight_replicate_node, 0_n},
                }},
                OpenDataflowEdge{
                    DataflowEdge{
                        DataflowOutput{n1_weight_replicate_node, 0_n},
                        DataflowInput{n1, 1_n},
                    },
                },
                OpenDataflowEdge{DataflowEdge{
                    DataflowOutput{n1, 0_n},
                    DataflowInput{n2, 0_n},
                }},
            },
            {i0},
            {{
                 OpenDataflowValue{i0},
                 i0_shape,
             },
             {
                 OpenDataflowValue{DataflowOutput{n1_weight_node, 0_n}},
                 lift_to_parallel(get_reduced_shape(n1_weight_shape)),
             },
             {
                 OpenDataflowValue{
                     DataflowOutput{n1_weight_replicate_node, 0_n}},
                 n1_weight_shape,
             },
             {
                 OpenDataflowValue{DataflowOutput{n1, 0_n}},
                 n1_output_shape,
             },
             {
                 OpenDataflowValue{DataflowOutput{n2, 0_n}},
                 n2_output_shape,
             }}};

    CHECK(result_data == correct_data);
  }
}
