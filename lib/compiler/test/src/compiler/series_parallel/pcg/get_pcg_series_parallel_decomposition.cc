#include "compiler/series_parallel/pcg/get_pcg_series_parallel_decomposition.h"
#include "op-attrs/ops/embedding.h"
#include "op-attrs/ops/linear.h"
#include "pcg/parallel_computation_graph/parallel_computation_graph.h"
#include "pcg/parallel_computation_graph/parallel_computation_graph_builder.h"
#include "utils/containers/get_only.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("get_pcg_series_parallel_decomposition(ParallelComputationGraph "
            "const &)") {
    SUBCASE("empty pcg") {
      ParallelComputationGraph pcg = empty_parallel_computation_graph();

      std::optional<SeriesParallelDecomposition> result =
          get_pcg_series_parallel_decomposition(pcg);
      // technically an empty graph is non-SP
      std::optional<SeriesParallelDecomposition> correct = std::nullopt;

      CHECK(result == correct);
    }

    TensorShape input_shape = TensorShape{TensorDims{FFOrdered<nonnegative_int>{
                                              10_n,
                                              12_n,
                                          }},
                                          DataType::FLOAT};
    InitializerAttrs zero_init = InitializerAttrs{ZeroInitializerAttrs{}};

    auto make_layer_attrs = [](auto const &op_attrs) -> ParallelLayerAttrs {
      return ParallelLayerAttrs{
          /*op_attrs=*/PCGOperatorAttrs{op_attrs},
          /*name=*/std::nullopt,
      };
    };

    SUBCASE("just a single input") {
      ParallelComputationGraph pcg = empty_parallel_computation_graph();

      ParallelLayerAddedResult added = pcg_add_input_layer(pcg, input_shape);
      parallel_layer_guid_t input_layer = added.parallel_layer;

      std::optional<SeriesParallelDecomposition> result =
          get_pcg_series_parallel_decomposition(pcg);
      std::optional<SeriesParallelDecomposition> correct =
          SeriesParallelDecomposition{input_layer.raw_graph_node};

      CHECK(result == correct);
    }

    SUBCASE("single operator plus inputs and weights") {
      ParallelComputationGraph pcg = empty_parallel_computation_graph();

      ParallelLayerAddedResult input_added =
          pcg_add_input_layer(pcg, input_shape);
      parallel_tensor_guid_t t_input = get_only(input_added.outputs);

      LinearAttrs linear_attrs = LinearAttrs{
          /*out_channels=*/14_n,
          /*use_bias=*/true,
          /*data_type=*/DataType::FLOAT,
          /*activation=*/Activation::RELU,
          /*regularizer=*/std::nullopt,
      };

      TensorShape projection_weights_shape =
          throw_if_unexpected(get_projection_shape(linear_attrs, input_shape));
      TensorShape bias_weights_shape =
          throw_if_unexpected(get_bias_shape(linear_attrs, input_shape));

      WeightAttrs projection_weight_attrs = WeightAttrs{
          /*shape=*/projection_weights_shape,
          /*initializer=*/InitializerAttrs{ZeroInitializerAttrs{}},
      };
      ParallelLayerAddedResult projection_weights_added = add_parallel_layer(
          pcg,
          /*layer_attrs=*/make_layer_attrs(projection_weight_attrs),
          /*inputs=*/{},
          /*weights=*/{});
      parallel_tensor_guid_t t_projection_weights =
          get_only(projection_weights_added.outputs);

      WeightAttrs bias_weight_attrs = WeightAttrs{
          /*shape=*/bias_weights_shape,
          /*initializer=*/InitializerAttrs{ZeroInitializerAttrs{}},
      };
      ParallelLayerAddedResult bias_weights_added = add_parallel_layer(
          pcg,
          /*layer_attrs=*/make_layer_attrs(bias_weight_attrs),
          /*inputs=*/{},
          /*weights=*/{});
      parallel_tensor_guid_t t_bias_weights =
          get_only(bias_weights_added.outputs);

      ParallelLayerAddedResult linear_added = add_parallel_layer(
          pcg,
          /*layer_attrs=*/make_layer_attrs(linear_attrs),
          /*inputs=*/{t_input},
          /*weights=*/{t_projection_weights, t_bias_weights});

      std::optional<SeriesParallelDecomposition> result =
          get_pcg_series_parallel_decomposition(pcg);
      std::optional<SeriesParallelDecomposition> correct =
          SeriesParallelDecomposition{SeriesSplit{{
              ParallelSplit{{
                  input_added.parallel_layer.raw_graph_node,
                  projection_weights_added.parallel_layer.raw_graph_node,
                  bias_weights_added.parallel_layer.raw_graph_node,
              }},
              linear_added.parallel_layer.raw_graph_node,
          }}};

      CHECK(result == correct);
    }

    SUBCASE("SP without weight nodes but non-SP with weight nodes (parallel op "
            "chain following is not necessary)") {
      // A minimal computation graph where without weights (w1 and w2) the
      // computation graph is series-parallel, but with weight nodes it is not,
      // but parallel op chain following is not necessary
      // (in this case because there are no parallel ops involved)
      //
      // w1   input   w2
      //  \   /   \   /
      //   op1     op2

      ParallelComputationGraph pcg = empty_parallel_computation_graph();

      InputAttrs input_attrs = InputAttrs{
          /*tensor_shape=*/input_shape,
      };

      LinearAttrs linear_attrs = LinearAttrs{
          /*out_channels=*/14_n,
          /*use_bias=*/false,
          /*data_type=*/DataType::FLOAT,
          /*activation=*/std::nullopt,
          /*regularizer=*/std::nullopt,
      };

      TensorShape projection_weight_shape =
          throw_if_unexpected(get_projection_shape(linear_attrs, input_shape));

      WeightAttrs projection_weight_attrs = WeightAttrs{
          /*tensor_shape=*/projection_weight_shape,
          /*initializer=*/zero_init,
      };

      ParallelLayerAddedResult input_added =
          add_parallel_layer(pcg, make_layer_attrs(input_attrs), {}, {});
      parallel_tensor_guid_t t_input = get_only(input_added.outputs);

      ParallelLayerAddedResult w1_added = add_parallel_layer(
          pcg, make_layer_attrs(projection_weight_attrs), {}, {});
      parallel_tensor_guid_t t_w1 = get_only(w1_added.outputs);

      ParallelLayerAddedResult w2_added = add_parallel_layer(
          pcg, make_layer_attrs(projection_weight_attrs), {}, {});
      parallel_tensor_guid_t t_w2 = get_only(w2_added.outputs);

      ParallelLayerAddedResult op1_added = add_parallel_layer(
          pcg, make_layer_attrs(linear_attrs), {t_input}, {t_w1});

      ParallelLayerAddedResult op2_added = add_parallel_layer(
          pcg, make_layer_attrs(linear_attrs), {t_input}, {t_w2});

      std::optional<SeriesParallelDecomposition> result =
          get_pcg_series_parallel_decomposition(pcg);
      std::optional<SeriesParallelDecomposition> correct =
          SeriesParallelDecomposition{SeriesSplit{{
              ParallelSplit{{
                  w1_added.parallel_layer.raw_graph_node,
                  input_added.parallel_layer.raw_graph_node,
                  w2_added.parallel_layer.raw_graph_node,
              }},
              ParallelSplit{{
                  op1_added.parallel_layer.raw_graph_node,
                  op2_added.parallel_layer.raw_graph_node,
              }},
          }}};

      CHECK(result == correct);
    }

    SUBCASE("SP without weight nodes but non-SP with weight node (parallel op "
            "chain following necessary)") {
      // A minimal computation graph where without weights (w1 and w2) the
      // computation graph is series-parallel, but with weight nodes it is not
      // and parallel op chain following is necessary
      //
      // w1   input   w2
      //  |    |       |
      //  |    p2     p4
      //  |    |       |
      // p1    p3     p5
      //  |    |       |
      //  |    |\     /
      //  |  op0 \    |
      //  \   /   |  /
      //   op1    op2

      ParallelComputationGraph pcg = empty_parallel_computation_graph();

      TensorShape input_shape = TensorShape{
          TensorDims{FFOrdered<nonnegative_int>{
              12_n,
              10_n,
          }},
          DataType::FLOAT,
      };

      ParallelLayerAddedResult input_added =
          pcg_add_input_layer(pcg, input_shape);
      parallel_layer_guid_t layer_input = input_added.parallel_layer;
      parallel_tensor_guid_t t_input = get_only(input_added.outputs);

      RepartitionAttrs p2_attrs = RepartitionAttrs{
          /*repartition_dim=*/ff_dim_t{0_n},
          /*repartition_degree=*/3_n,
      };
      ParallelLayerAddedResult p2_added =
          add_parallel_layer(pcg, make_layer_attrs(p2_attrs), {t_input}, {});
      parallel_tensor_guid_t t_p2 = get_only(p2_added.outputs);

      ParallelLayerAttrs p3_attrs = ParallelLayerAttrs{
          PCGOperatorAttrs{RepartitionAttrs{
              /*repartition_dim=*/ff_dim_t{1_n},
              /*repartition_degree=*/2_n,
          }},
          /*name=*/std::nullopt,
      };
      ParallelLayerAddedResult p3_added =
          add_parallel_layer(pcg, p3_attrs, {t_p2}, {});
      parallel_tensor_guid_t t_p3 = get_only(p3_added.outputs);

      CastAttrs op0_attrs = CastAttrs{
          /*dtype=*/DataType::INT32,
      };
      ParallelLayerAddedResult op0_added =
          add_parallel_layer(pcg, make_layer_attrs(op0_attrs), {t_p3}, {});
      parallel_tensor_guid_t t_op0 = get_only(op0_added.outputs);

      EmbeddingAttrs op1_attrs = EmbeddingAttrs{
          /*num_entires=*/100_n,
          /*out_channels=*/22_n,
          /*aggr=*/AggregateOp::SUM,
          /*data_type=*/DataType::FLOAT,
      };

      TensorShape casted_input_shape =
          get_reduced_shape(get_parallel_tensor_shape(pcg, t_op0));

      WeightAttrs w1_attrs = WeightAttrs{
          /*tensor_shape=*/throw_if_unexpected(
              get_weights_shape(op1_attrs, casted_input_shape)),
          /*initializer=*/InitializerAttrs{ZeroInitializerAttrs{}},
      };
      ParallelLayerAddedResult w1_added =
          add_parallel_layer(pcg, make_layer_attrs(w1_attrs), {}, {});
      parallel_tensor_guid_t t_w1 = get_only(w1_added.outputs);

      ReplicateAttrs p1_attrs = ReplicateAttrs{
          /*replicate_degree=*/6_n,
      };
      ParallelLayerAddedResult p1_added =
          add_parallel_layer(pcg, make_layer_attrs(p1_attrs), {t_w1}, {});
      parallel_tensor_guid_t t_p1 = get_only(p1_added.outputs);

      ParallelLayerAddedResult op1_added =
          add_parallel_layer(pcg, make_layer_attrs(op1_attrs), {t_op0}, {t_p1});

      LinearAttrs op2_attrs = LinearAttrs{
          /*out_channels=*/14_n,
          /*use_bias=*/false,
          /*data_type=*/DataType::FLOAT,
          /*activation=*/std::nullopt,
          /*regularizer=*/std::nullopt,
      };

      WeightAttrs w2_attrs = WeightAttrs{
          /*tensor_shape=*/throw_if_unexpected(
              get_projection_shape(op2_attrs, input_shape)),
          /*initializer=*/InitializerAttrs{ZeroInitializerAttrs{}},
      };
      ParallelLayerAddedResult w2_added =
          add_parallel_layer(pcg, make_layer_attrs(w2_attrs), {}, {});
      parallel_tensor_guid_t t_w2 = get_only(w2_added.outputs);

      ReplicateAttrs p4_attrs = ReplicateAttrs{
          /*replicate_degree=*/3_n,
      };
      ParallelLayerAddedResult p4_added =
          add_parallel_layer(pcg, make_layer_attrs(p4_attrs), {t_w2}, {});
      parallel_tensor_guid_t t_p4 = get_only(p4_added.outputs);

      RepartitionAttrs p5_attrs = RepartitionAttrs{
          /*repartition_dim=*/ff_dim_t{0_n},
          /*repartition_degree=*/2_n,
      };
      ParallelLayerAddedResult p5_added =
          add_parallel_layer(pcg, make_layer_attrs(p5_attrs), {t_p4}, {});
      parallel_tensor_guid_t t_p5 = get_only(p5_added.outputs);

      ParallelLayerAddedResult op2_added =
          add_parallel_layer(pcg, make_layer_attrs(op2_attrs), {t_p3}, {t_p5});

      std::optional<SeriesParallelDecomposition> result =
          get_pcg_series_parallel_decomposition(pcg);
      std::optional<SeriesParallelDecomposition> correct =
          SeriesParallelDecomposition{SeriesSplit{{
              ParallelSplit{{
                  SeriesSplit{{
                      w1_added.parallel_layer.raw_graph_node,
                      p1_added.parallel_layer.raw_graph_node,
                  }},
                  SeriesSplit{{
                      input_added.parallel_layer.raw_graph_node,
                      p2_added.parallel_layer.raw_graph_node,
                      p3_added.parallel_layer.raw_graph_node,
                  }},
                  SeriesSplit{{
                      w2_added.parallel_layer.raw_graph_node,
                      p4_added.parallel_layer.raw_graph_node,
                      p5_added.parallel_layer.raw_graph_node,
                  }},
              }},
              ParallelSplit{{
                  SeriesSplit{{
                      op0_added.parallel_layer.raw_graph_node,
                      op1_added.parallel_layer.raw_graph_node,
                  }},
                  op2_added.parallel_layer.raw_graph_node,
              }},
          }}};

      CHECK(result == correct);
    }

    ParallelComputationGraph pcg = empty_parallel_computation_graph();
    InputAttrs input_attrs = InputAttrs{
        /*tensor_shape=*/input_shape,
    };
    ElementUnaryAttrs relu_attrs = ElementUnaryAttrs{
        /*op_type=*/OperatorType::RELU,
        /*scalar=*/std::nullopt,
    };

    SUBCASE("SP with or without preprocessing, but preprocessing would change "
            "resulting SP "
            "decomposition") {
      // parallel computation graph:
      //
      //  input1   input2
      //    |        |
      //   op1      op2

      ParallelLayerAddedResult input1_added =
          add_parallel_layer(pcg, make_layer_attrs(input_attrs), {}, {});
      parallel_tensor_guid_t t_input1 = get_only(input1_added.outputs);

      ParallelLayerAddedResult input2_added =
          add_parallel_layer(pcg, make_layer_attrs(input_attrs), {}, {});
      parallel_tensor_guid_t t_input2 = get_only(input2_added.outputs);

      ParallelLayerAddedResult op1_added =
          add_parallel_layer(pcg, make_layer_attrs(relu_attrs), {t_input1}, {});

      ParallelLayerAddedResult op2_added =
          add_parallel_layer(pcg, make_layer_attrs(relu_attrs), {t_input2}, {});

      std::optional<SeriesParallelDecomposition> result =
          get_pcg_series_parallel_decomposition(pcg);
      std::optional<SeriesParallelDecomposition> correct =
          SeriesParallelDecomposition{ParallelSplit{{
              SeriesSplit{{
                  input1_added.parallel_layer.raw_graph_node,
                  op1_added.parallel_layer.raw_graph_node,
              }},
              SeriesSplit{{
                  input2_added.parallel_layer.raw_graph_node,
                  op2_added.parallel_layer.raw_graph_node,
              }},
          }}};

      CHECK(result == correct);
    }

    SUBCASE("not SP with or without weight nodes") {
      // parallel computation graph:
      //
      //    input1
      //     /  \
      //   op1  op2
      //    | \  |
      //    |  \ |
      //   op3  op4

      ParallelLayerAddedResult input1_added =
          add_parallel_layer(pcg, make_layer_attrs(input_attrs), {}, {});
      parallel_tensor_guid_t t_input1 = get_only(input1_added.outputs);

      ElementBinaryAttrs ew_add_attrs = ElementBinaryAttrs{
          /*type=*/OperatorType::EW_ADD,
          /*compute_type=*/DataType::FLOAT,
          /*should_broadcast_lhs=*/false,
          /*should_broadcast_rhs=*/false,
      };

      ParallelLayerAddedResult op1_added =
          add_parallel_layer(pcg, make_layer_attrs(relu_attrs), {t_input1}, {});
      parallel_tensor_guid_t t_op1 = get_only(op1_added.outputs);

      ParallelLayerAddedResult op2_added =
          add_parallel_layer(pcg, make_layer_attrs(relu_attrs), {t_input1}, {});
      parallel_tensor_guid_t t_op2 = get_only(op2_added.outputs);

      ParallelLayerAddedResult op3_added =
          add_parallel_layer(pcg, make_layer_attrs(relu_attrs), {t_op1}, {});

      ParallelLayerAddedResult op4_added = add_parallel_layer(
          pcg, make_layer_attrs(ew_add_attrs), {t_op1, t_op2}, {});

      std::optional<SeriesParallelDecomposition> result =
          get_pcg_series_parallel_decomposition(pcg);
      std::optional<SeriesParallelDecomposition> correct = std::nullopt;

      CHECK(result == correct);
    }
  }
}
