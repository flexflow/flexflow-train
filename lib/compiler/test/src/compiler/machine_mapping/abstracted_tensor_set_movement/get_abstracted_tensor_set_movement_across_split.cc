#include "compiler/machine_mapping/abstracted_tensor_set_movement/get_abstracted_tensor_set_movement_across_split.h"
#include "compiler/machine_mapping/transitive_reduced_pcg.h"
#include "pcg/parallel_computation_graph/parallel_computation_graph.h"
#include "utils/containers/get_only.h"
#include "utils/full_binary_tree/binary_tree_path.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("get_abstracted_tensor_set_movement_across_split") {
    auto make_series_split = [](PCGBinarySPDecomposition const &lhs,
                                PCGBinarySPDecomposition const &rhs) {
      return PCGBinarySPDecomposition{PCGBinarySeriesSplit{lhs, rhs}};
    };

    auto make_parallel_split = [](PCGBinarySPDecomposition const &lhs,
                                  PCGBinarySPDecomposition const &rhs) {
      return PCGBinarySPDecomposition{PCGBinaryParallelSplit{lhs, rhs}};
    };

    auto make_leaf = [](parallel_layer_guid_t const &l) {
      return PCGBinarySPDecomposition{l};
    };

    ParallelComputationGraph pcg = empty_parallel_computation_graph();

    ParallelTensorShape input_shape = ParallelTensorShape{
        ParallelTensorDims{
            FFOrdered<ShardParallelDim>{
                ShardParallelDim{10_n, 2_n},
                ShardParallelDim{12_n, 1_n},
            },
            ReplicaParallelDimSet{
                SumDegree{1_n},
                DiscardCopyDegree{1_n},
            },
        },
        DataType::FLOAT,
    };
    ParallelLayerAttrs relu_attrs = ParallelLayerAttrs{
        /*op_attrs=*/PCGOperatorAttrs{
            ElementUnaryAttrs{
                /*op_type=*/OperatorType::RELU,
                /*scalar=*/std::nullopt,
            },
        },
        /*name=*/std::nullopt,
    };

    ParallelLayerAttrs ew_add_attrs = ParallelLayerAttrs{
        /*op_attrs=*/PCGOperatorAttrs{
            ElementBinaryAttrs{
                /*type=*/OperatorType::EW_ADD,
                /*compute_type=*/DataType::FLOAT,
                /*should_broadcast_lhs=*/false,
                /*should_broadcast_rhs=*/false,
            },
        },
        /*name=*/std::nullopt,
    };

    ParallelTensorAttrs relu_output_attrs = ParallelTensorAttrs{
        /*shape=*/input_shape,
        /*sync_type=*/std::nullopt,
        /*initializer=*/std::nullopt,
        /*create_gradients=*/CreateGrad::YES,
    };

    SUBCASE("no edges across split") {
      ParallelLayerAddedResult input1 = pcg_add_input_layer(pcg, input_shape);
      ParallelLayerAddedResult input2 = pcg_add_input_layer(pcg, input_shape);

      PCGBinarySeriesSplit split = PCGBinarySeriesSplit{
          make_leaf(input1.parallel_layer),
          make_leaf(input2.parallel_layer),
      };

      AbstractedTensorSetMovement result =
          get_abstracted_tensor_set_movement_across_split(
              pcg_get_transitive_reduction(pcg), split);

      AbstractedTensorSetMovement correct = AbstractedTensorSetMovement{
          /*single_tensor_movements=*/{},
      };

      CHECK(result == correct);
    }

    SUBCASE("single edge across split") {
      ParallelLayerAddedResult input = pcg_add_input_layer(pcg, input_shape);

      ParallelLayerAddedResult layer_1 = add_parallel_layer(
          pcg, relu_attrs, {get_only(input.outputs)}, {relu_output_attrs});
      ParallelLayerAddedResult layer_2 = add_parallel_layer(
          pcg, relu_attrs, {get_only(layer_1.outputs)}, {relu_output_attrs});

      PCGBinarySeriesSplit split = PCGBinarySeriesSplit{
          make_series_split(make_leaf(input.parallel_layer),
                            make_leaf(layer_1.parallel_layer)),
          make_leaf(layer_2.parallel_layer),
      };

      AbstractedTensorSetMovement result =
          get_abstracted_tensor_set_movement_across_split(
              pcg_get_transitive_reduction(pcg), split);

      AbstractedTensorSetMovement correct = AbstractedTensorSetMovement{
          /*single_tensor_movements=*/{
              AbstractedSingleTensorMovement{
                  /*parallel_tensor_shape=*/input_shape,
                  /*src_machine_views=*/
                  {
                      BinaryTreePath{{
                          BinaryTreePathEntry::RIGHT_CHILD,
                      }},
                  },
                  /*dst_machine_views=*/
                  {
                      BinaryTreePath{{}},
                  },
              },
          },
      };

      CHECK(result == correct);
    }

    SUBCASE("does not include edges removed by transitive reduction") {
      ParallelLayerAddedResult input = pcg_add_input_layer(pcg, input_shape);

      ParallelLayerAddedResult layer_1 = add_parallel_layer(
          pcg, relu_attrs, {get_only(input.outputs)}, {relu_output_attrs});

      ParallelLayerAddedResult layer_2 = add_parallel_layer(
          pcg, relu_attrs, {get_only(layer_1.outputs)}, {relu_output_attrs});

      ParallelLayerAddedResult layer_3 = add_parallel_layer(
          pcg,
          ew_add_attrs,
          {get_only(layer_1.outputs), get_only(layer_2.outputs)},
          {relu_output_attrs});

      PCGBinarySeriesSplit split = PCGBinarySeriesSplit{
          make_series_split(
              make_leaf(input.parallel_layer),
              make_series_split(make_leaf(layer_1.parallel_layer),
                                make_leaf(layer_2.parallel_layer))),
          make_leaf(layer_3.parallel_layer),
      };

      AbstractedTensorSetMovement result =
          get_abstracted_tensor_set_movement_across_split(
              pcg_get_transitive_reduction(pcg), split);

      AbstractedTensorSetMovement correct = AbstractedTensorSetMovement{
          /*single_tensor_movements=*/{
              AbstractedSingleTensorMovement{
                  /*parallel_tensor_shape=*/input_shape,
                  /*src_machine_views=*/
                  {
                      BinaryTreePath{{
                          BinaryTreePathEntry::RIGHT_CHILD,
                          BinaryTreePathEntry::RIGHT_CHILD,
                      }},
                  },
                  /*dst_machine_views=*/
                  {
                      BinaryTreePath{{}},
                  },
              },
          },
      };

      CHECK(result == correct);
    }

    SUBCASE("single tensor, multiple consumers across split") {
      ParallelLayerAddedResult input = pcg_add_input_layer(pcg, input_shape);

      ParallelLayerAddedResult layer_1 = add_parallel_layer(
          pcg, relu_attrs, {get_only(input.outputs)}, {relu_output_attrs});

      ParallelLayerAddedResult layer_2 = add_parallel_layer(
          pcg, relu_attrs, {get_only(layer_1.outputs)}, {relu_output_attrs});

      ParallelLayerAddedResult layer_3 = add_parallel_layer(
          pcg, relu_attrs, {get_only(layer_1.outputs)}, {relu_output_attrs});

      PCGBinarySeriesSplit split = PCGBinarySeriesSplit{
          make_series_split(make_leaf(input.parallel_layer),
                            make_leaf(layer_1.parallel_layer)),
          make_parallel_split(make_leaf(layer_2.parallel_layer),
                              make_leaf(layer_3.parallel_layer)),
      };

      AbstractedTensorSetMovement result =
          get_abstracted_tensor_set_movement_across_split(
              pcg_get_transitive_reduction(pcg), split);

      AbstractedTensorSetMovement correct = AbstractedTensorSetMovement{
          /*single_tensor_movements=*/{
              AbstractedSingleTensorMovement{
                  /*parallel_tensor_shape=*/input_shape,
                  /*src_machine_views=*/
                  {
                      BinaryTreePath{{
                          BinaryTreePathEntry::RIGHT_CHILD,
                      }},
                  },
                  /*dst_machine_views=*/
                  {
                      BinaryTreePath{{
                          BinaryTreePathEntry::LEFT_CHILD,
                      }},
                      BinaryTreePath{{
                          BinaryTreePathEntry::RIGHT_CHILD,
                      }},
                  },
              },
          },
      };

      CHECK(result == correct);
    }

    SUBCASE("multiple tensors, multiple consumers across split") {
      ParallelLayerAddedResult input = pcg_add_input_layer(pcg, input_shape);

      ParallelLayerAddedResult layer_1 = add_parallel_layer(
          pcg, relu_attrs, {get_only(input.outputs)}, {relu_output_attrs});

      ParallelLayerAddedResult layer_2 = add_parallel_layer(
          pcg, relu_attrs, {get_only(input.outputs)}, {relu_output_attrs});

      ParallelLayerAddedResult layer_3 = add_parallel_layer(
          pcg, relu_attrs, {get_only(layer_1.outputs)}, {relu_output_attrs});

      ParallelLayerAddedResult layer_4 = add_parallel_layer(
          pcg,
          ew_add_attrs,
          {get_only(layer_1.outputs), get_only(layer_2.outputs)},
          {relu_output_attrs});

      PCGBinarySeriesSplit split = PCGBinarySeriesSplit{
          make_series_split(
              make_leaf(input.parallel_layer),
              make_parallel_split(make_leaf(layer_1.parallel_layer),
                                  make_leaf(layer_2.parallel_layer))),
          make_parallel_split(make_leaf(layer_3.parallel_layer),
                              make_leaf(layer_4.parallel_layer))};

      AbstractedTensorSetMovement result =
          get_abstracted_tensor_set_movement_across_split(
              pcg_get_transitive_reduction(pcg), split);

      AbstractedTensorSetMovement correct = AbstractedTensorSetMovement{
          /*single_tensor_movements=*/{
              AbstractedSingleTensorMovement{
                  /*parallel_tensor_shape=*/input_shape,
                  /*src_machine_views=*/
                  {
                      BinaryTreePath{{
                          BinaryTreePathEntry::RIGHT_CHILD,
                          BinaryTreePathEntry::LEFT_CHILD,
                      }},
                  },
                  /*dst_machine_views=*/
                  {
                      BinaryTreePath{{
                          BinaryTreePathEntry::LEFT_CHILD,
                      }},
                      BinaryTreePath{{
                          BinaryTreePathEntry::RIGHT_CHILD,
                      }},
                  },
              },
              AbstractedSingleTensorMovement{
                  /*parallel_tensor_shape=*/input_shape,
                  /*src_machine_views=*/
                  {
                      BinaryTreePath{{
                          BinaryTreePathEntry::RIGHT_CHILD,
                          BinaryTreePathEntry::RIGHT_CHILD,
                      }},
                  },
                  /*dst_machine_views=*/
                  {
                      BinaryTreePath{{
                          BinaryTreePathEntry::RIGHT_CHILD,
                      }},
                  },
              },
          },
      };

      CHECK(result == correct);
    }
  }
}
