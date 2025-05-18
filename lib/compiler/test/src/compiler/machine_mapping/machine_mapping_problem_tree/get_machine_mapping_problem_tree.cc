#include "compiler/machine_mapping/machine_mapping_problem_tree/get_machine_mapping_problem_tree.h"
#include "compiler/machine_mapping/machine_mapping_problem_tree/machine_mapping_problem_tree.h"
#include "compiler/series_parallel/pcg/get_pcg_balanced_binary_sp_decomposition.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "pcg/computation_graph_builder.h"
#include "pcg/operator_task_space.h"
#include "pcg/parallel_computation_graph/parallel_computation_graph.h"
#include "pcg/parallel_computation_graph/parallel_computation_graph_builder.h"
#include "pcg/pcg_from_computation_graph.h"
#include "utils/containers/extend.h"
#include "utils/containers/get_only.h"
#include "utils/containers/vector_of.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("get_machine_mapping_problem_tree") {
    auto pcg_make_leaf = [](parallel_layer_guid_t const &l) {
      return PCGBinarySPDecomposition{l};
    };

    auto pcg_make_series = [](PCGBinarySPDecomposition const &lhs,
                              PCGBinarySPDecomposition const &rhs) {
      return PCGBinarySPDecomposition{
          PCGBinarySeriesSplit{
              lhs,
              rhs,
          },
      };
    };

    auto pcg_make_parallel = [](PCGBinarySPDecomposition const &lhs,
                                PCGBinarySPDecomposition const &rhs) {
      return PCGBinarySPDecomposition{
          PCGBinaryParallelSplit{
              lhs,
              rhs,
          },
      };
    };

    auto mm_problem_tree_make_leaf = [](UnmappedOpCostEstimateKey const &k) {
      return MachineMappingProblemTree{k};
    };

    auto mm_problem_tree_make_series =
        [](AbstractedTensorSetMovement const &tensor_set_movement,
           MachineMappingProblemTree const &lhs,
           MachineMappingProblemTree const &rhs) {
          return MachineMappingProblemTree{
              MMProblemTreeSeriesSplit{
                  tensor_set_movement,
                  lhs,
                  rhs,
              },
          };
        };

    auto mm_problem_tree_make_parallel =
        [](MachineMappingProblemTree const &lhs,
           MachineMappingProblemTree const &rhs) {
          return MachineMappingProblemTree{
              MMProblemTreeParallelSplit{
                  lhs,
                  rhs,
              },
          };
        };

    ParallelComputationGraph pcg = empty_parallel_computation_graph();

    TensorShape input_shape = TensorShape{
        TensorDims{
            FFOrdered<nonnegative_int>{
                10_n,
                1_n,
            },
        },
        DataType::FLOAT,
    };
    ParallelTensorShape par_input_shape = lift_to_parallel(input_shape);

    auto make_output_attrs = [](ParallelTensorShape const &shape) {
      return ParallelTensorAttrs{
          /*shape=*/shape,
          /*create_gradients=*/CreateGrad::YES,
      };
    };

    auto make_layer_attrs = [](PCGOperatorAttrs const &op_attrs) {
      return ParallelLayerAttrs{
          /*op_attrs=*/op_attrs,
          /*name=*/std::nullopt,
      };
    };

    PCGOperatorAttrs input_attrs = PCGOperatorAttrs{InputAttrs{input_shape}};

    auto make_operator_task_space = [&](ParallelTensorShape const &shape) {
      std::vector<nonnegative_int> degrees;
      extend(degrees, vector_of(ff_ordered_shard_degrees(shape)));
      degrees.push_back(get_sum_degree(shape));
      degrees.push_back(get_discard_copy_degree(shape));
      return OperatorTaskSpace{degrees};
    };

    auto make_input_key =
        [&](ParallelTensorShape const &parallel_tensor_shape) {
          return UnmappedOpCostEstimateKey{
              /*op_attrs=*/input_attrs,
              /*input_shapes=*/{},
              /*weight_shapes=*/{},
              /*output_shapes=*/{parallel_tensor_shape},
              /*op_task_space=*/make_operator_task_space(parallel_tensor_shape),
          };
        };

    SUBCASE("single layer") {
      ParallelLayerAddedResult input_added =
          add_parallel_layer(pcg,
                             /*layer_attrs=*/make_layer_attrs(input_attrs),
                             /*inputs=*/{},
                             /*output_labels=*/{});
      parallel_layer_guid_t input_layer = input_added.parallel_layer;

      UnmappedOpCostEstimateKey input_key = make_input_key(par_input_shape);

      PCGBinarySPDecomposition sp_decomposition =
          PCGBinarySPDecomposition{input_layer};

      MachineMappingProblemTree result =
          get_machine_mapping_problem_tree(pcg, sp_decomposition);
      MachineMappingProblemTree correct = MachineMappingProblemTree{input_key};

      CHECK(result == correct);
    }

    SUBCASE("two layers in series") {
      ParallelLayerAddedResult input_added =
          add_parallel_layer(pcg,
                             /*layer_attrs=*/make_layer_attrs(input_attrs),
                             /*inputs=*/{},
                             /*output_labels=*/{});
      parallel_layer_guid_t input_layer = input_added.parallel_layer;
      parallel_tensor_guid_t input = get_only(input_added.outputs);

      UnmappedOpCostEstimateKey input_key = make_input_key(par_input_shape);

      PCGOperatorAttrs relu_attrs = PCGOperatorAttrs{
          ElementUnaryAttrs{
              /*op_type=*/OperatorType::RELU,
              /*scalar=*/std::nullopt,
          },
      };
      ParallelTensorShape relu_output_shape = par_input_shape;
      ParallelLayerAddedResult relu_added =
          add_parallel_layer(pcg, make_layer_attrs(relu_attrs), {input}, {});
      parallel_layer_guid_t relu_layer = relu_added.parallel_layer;
      parallel_tensor_guid_t relu_output = get_only(relu_added.outputs);

      OperatorTaskSpace relu_task_space =
          get_operator_task_space(pcg, relu_layer);

      UnmappedOpCostEstimateKey relu_key = UnmappedOpCostEstimateKey{
          /*op_attrs=*/relu_attrs,
          /*input_shapes=*/{par_input_shape},
          /*weight_shapes=*/{},
          /*output_shapes=*/{relu_output_shape},
          /*op_task_space=*/relu_task_space,
      };

      PCGBinarySPDecomposition sp_decomposition = pcg_make_series(
          pcg_make_leaf(input_layer), pcg_make_leaf(relu_layer));

      MachineMappingProblemTree result =
          get_machine_mapping_problem_tree(pcg, sp_decomposition);

      MachineMappingProblemTree correct = mm_problem_tree_make_series(
          AbstractedTensorSetMovement{{
              AbstractedSingleTensorMovement{
                  /*parallel_tensor_shape=*/par_input_shape,
                  /*src_machine_views=*/
                  {
                      BinaryTreePath{{}},
                  },
                  /*dst_machine_views=*/
                  {
                      BinaryTreePath{{}},
                  },
              },
          }},
          mm_problem_tree_make_leaf(input_key),
          mm_problem_tree_make_leaf(relu_key));

      CHECK(result == correct);
    }

    SUBCASE("two layers in parallel") {
      ParallelLayerAddedResult input1_added =
          pcg_add_input_layer(pcg, input_shape);
      parallel_layer_guid_t input1_layer = input1_added.parallel_layer;
      UnmappedOpCostEstimateKey input1_key = make_input_key(par_input_shape);

      ParallelLayerAddedResult input2_added =
          pcg_add_input_layer(pcg, input_shape);
      parallel_layer_guid_t input2_layer = input2_added.parallel_layer;
      UnmappedOpCostEstimateKey input2_key = make_input_key(par_input_shape);

      PCGBinarySPDecomposition sp_decomposition = pcg_make_parallel(
          pcg_make_leaf(input1_layer), pcg_make_leaf(input2_layer));

      MachineMappingProblemTree result =
          get_machine_mapping_problem_tree(pcg, sp_decomposition);

      MachineMappingProblemTree correct =
          mm_problem_tree_make_parallel(mm_problem_tree_make_leaf(input1_key),
                                        mm_problem_tree_make_leaf(input2_key));

      CHECK(result == correct);
    }

    SUBCASE("multiple tensors across split") {
      ParallelLayerAddedResult input1_added =
          pcg_add_input_layer(pcg, input_shape);
      parallel_layer_guid_t input1_layer = input1_added.parallel_layer;
      parallel_tensor_guid_t input1_tensor = get_only(input1_added.outputs);
      UnmappedOpCostEstimateKey input1_key = make_input_key(par_input_shape);

      ParallelLayerAddedResult input2_added =
          pcg_add_input_layer(pcg, input_shape);
      parallel_layer_guid_t input2_layer = input2_added.parallel_layer;
      parallel_tensor_guid_t input2_tensor = get_only(input2_added.outputs);
      UnmappedOpCostEstimateKey input2_key = make_input_key(par_input_shape);

      PCGOperatorAttrs ew_op_attrs = PCGOperatorAttrs{
          ElementBinaryAttrs{
              /*type=*/OperatorType::EW_ADD,
              /*compute_type=*/DataType::FLOAT,
              /*should_broadcast_lhs=*/false,
              /*should_broadcast_rhs=*/false,
          },
      };
      ParallelTensorShape ew_op_output_shape = par_input_shape;
      ParallelLayerAddedResult ew_op_added =
          add_parallel_layer(pcg,
                             make_layer_attrs(ew_op_attrs),
                             {input1_tensor, input2_tensor},
                             {});
      parallel_layer_guid_t ew_op_layer = ew_op_added.parallel_layer;
      OperatorTaskSpace ew_op_task_space =
          get_operator_task_space(pcg, ew_op_layer);
      UnmappedOpCostEstimateKey ew_op_key = UnmappedOpCostEstimateKey{
          /*op_attrs=*/ew_op_attrs,
          /*input_shapes=*/{par_input_shape, par_input_shape},
          /*weight_shapes=*/{},
          /*output_shapes=*/{ew_op_output_shape},
          /*op_task_space=*/ew_op_task_space,
      };

      PCGBinarySPDecomposition sp_decomposition =
          pcg_make_series(pcg_make_parallel(pcg_make_leaf(input1_layer),
                                            pcg_make_leaf(input2_layer)),
                          pcg_make_leaf(ew_op_layer));

      MachineMappingProblemTree result =
          get_machine_mapping_problem_tree(pcg, sp_decomposition);

      MachineMappingProblemTree correct = mm_problem_tree_make_series(
          AbstractedTensorSetMovement{{
              AbstractedSingleTensorMovement{
                  /*parallel_tensor_shape=*/par_input_shape,
                  /*src_machine_views=*/
                  {
                      BinaryTreePath{{
                          BinaryTreePathEntry::LEFT_CHILD,
                      }},
                  },
                  /*dst_machine_views=*/
                  {
                      BinaryTreePath{{}},
                  },
              },
              AbstractedSingleTensorMovement{
                  /*parallel_tensor_shape=*/par_input_shape,
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
          }},
          /*pre=*/
          mm_problem_tree_make_parallel(mm_problem_tree_make_leaf(input1_key),
                                        mm_problem_tree_make_leaf(input2_key)),
          /*post=*/mm_problem_tree_make_leaf(ew_op_key));

      CHECK(result == correct);
    }
  }

  TEST_CASE("from pcg") {
    ComputationGraph cg = [&] {
      ComputationGraphBuilder b;
      TensorShape input_tensor_shape = TensorShape{
          TensorDims{
              FFOrdered<nonnegative_int>{nonnegative_int{32},
                                         nonnegative_int{64}},
          },
          DataType::FLOAT,
      };
      tensor_guid_t t = b.create_input(input_tensor_shape, CreateGrad::YES);
      t = b.dense(t,
                  /*outDim=*/nonnegative_int{16},
                  /*activation=*/std::nullopt);
      t = b.gelu(t);
      t = b.dense(t,
                  /*outDim=*/nonnegative_int{12},
                  /*activation=*/std::nullopt,
                  /*use_bias=*/false,
                  /*data_type=*/DataType::FLOAT,
                  /*kernel_initializer=*/std::nullopt,
                  /*bias_initializer=*/std::nullopt);
      t = b.relu(t);
      t = b.dense(t,
                  /*outDim=*/nonnegative_int{8},
                  /*activation=*/Activation::RELU);
      return b.computation_graph;
    }();

    ParallelComputationGraph pcg = pcg_from_computation_graph(cg);

    PCGBinarySPDecomposition sp_decomp =
        expect(get_pcg_balanced_binary_sp_decomposition(pcg),
               "Failed to get SP decomposition of PCG");

    MachineMappingProblemTree problem_tree =
        get_machine_mapping_problem_tree(pcg, sp_decomp);
  }
}
