#include "compiler/machine_mapping/get_tensor_set_movement_across_split.h"
#include "../cost_estimator_for_test.h"
#include "compiler/machine_mapping/transitive_reduced_pcg.h"
#include "pcg/machine_view.h"
#include "pcg/parallel_computation_graph/parallel_computation_graph.h"
#include "pcg/parallel_computation_graph/parallel_computation_graph_builder.h"
#include "utils/containers/get_only.h"
#include <doctest/doctest.h>
#include <fstream>

using namespace ::FlexFlow;

bool isDebuggerActive() {
  std::ifstream in("/proc/self/status");
  for (std::string line; std::getline(in, line);) {
    static int const PREFIX_LEN = 11;
    if (line.compare(0, PREFIX_LEN, "TracerPid:\t") == 0) {
      return line.length() > PREFIX_LEN && line[PREFIX_LEN] != '0';
    }
  }
  return false;
}

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("get_tensor_set_movement_across_split") {
    auto make_pcg_series_split = [](PCGBinarySPDecomposition const &lhs,
                                    PCGBinarySPDecomposition const &rhs) {
      return PCGBinarySPDecomposition{PCGBinarySeriesSplit{lhs, rhs}};
    };

    auto make_pcg_parallel_split = [](PCGBinarySPDecomposition const &lhs,
                                      PCGBinarySPDecomposition const &rhs) {
      return PCGBinarySPDecomposition{PCGBinaryParallelSplit{lhs, rhs}};
    };

    auto make_pcg_leaf_node = [](parallel_layer_guid_t const &l) {
      return PCGBinarySPDecomposition{l};
    };

    ParallelComputationGraph pcg = empty_parallel_computation_graph();

    TensorShape input_shape = TensorShape{
        TensorDims{
            FFOrdered<nonnegative_int>{
                10_n,
                12_n,
            },
        },
        DataType::FLOAT,
    };

    ParallelLayerAddedResult input = pcg_add_input_layer(pcg, input_shape);
    parallel_tensor_guid_t t_input = get_only(input.outputs);

    ParallelLayerAttrs partition_attrs = ParallelLayerAttrs{
        /*op_attrs=*/PCGOperatorAttrs{
            RepartitionAttrs{
                /*repartition_dim=*/ff_dim_t{0_n},
                /*repartition_degree=*/2_n,
            },
        },
        /*name=*/std::nullopt,
    };
    ParallelLayerAddedResult partition_input =
        add_parallel_layer(pcg, partition_attrs, {t_input}, {});
    parallel_tensor_guid_t t_partition_input =
        get_only(partition_input.outputs);

    ParallelTensorShape partitioned_input_shape =
        get_parallel_tensor_shape(pcg, t_partition_input);

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

    ParallelLayerAddedResult relu_1 =
        add_parallel_layer(pcg, relu_attrs, {t_partition_input}, {});
    parallel_tensor_guid_t t_relu_1 = get_only(relu_1.outputs);
    ParallelLayerAddedResult relu_2 =
        add_parallel_layer(pcg, relu_attrs, {t_relu_1}, {});

    MachineView pre_mv1 = MachineView{
        /*start=*/MachineSpaceCoordinate{
            /*node_idx=*/0_n,
            /*device_idx=*/0_n,
            /*device_type=*/DeviceType::GPU,
        },
        /*dimensions=*/
        {
            MachineViewDimension{
                stride_t{1_n},
                MachineSpecificationDimension::INTRA_NODE,
            },
        },
    };

    MachineView pre_mv2 = MachineView{
        /*start=*/MachineSpaceCoordinate{
            /*node_idx=*/0_n,
            /*device_idx=*/0_n,
            /*device_type=*/DeviceType::GPU,
        },
        /*dimensions=*/
        {
            MachineViewDimension{
                stride_t{2_n},
                MachineSpecificationDimension::INTRA_NODE,
            },
        },
    };

    MachineView post_mv1 = MachineView{
        /*start=*/MachineSpaceCoordinate{
            /*node_idx=*/0_n,
            /*device_idx=*/0_n,
            /*device_type=*/DeviceType::GPU,
        },
        /*dimensions=*/
        {
            MachineViewDimension{
                stride_t{3_n},
                MachineSpecificationDimension::INTRA_NODE,
            },
        },
    };

    MachineView post_mv2 = MachineView{
        /*start=*/MachineSpaceCoordinate{
            /*node_idx=*/0_n,
            /*device_idx=*/0_n,
            /*device_type=*/DeviceType::GPU,
        },
        /*dimensions=*/
        {
            MachineViewDimension{
                stride_t{4_n},
                MachineSpecificationDimension::INTRA_NODE,
            },
        },
    };

    SUBCASE("single edge across split") {
      PCGBinarySeriesSplit split = PCGBinarySeriesSplit{
          make_pcg_series_split(
              make_pcg_series_split(
                  make_pcg_leaf_node(input.parallel_layer),
                  make_pcg_leaf_node(partition_input.parallel_layer)),
              make_pcg_leaf_node(relu_1.parallel_layer)),
          make_pcg_leaf_node(relu_2.parallel_layer),
      };

      auto pre_mapping = ParallelLayerGuidObliviousMachineMapping{{
          {BinaryTreePath{{
               BinaryTreePathEntry::RIGHT_CHILD,
           }},
           pre_mv1},
      }};

      auto post_mapping = ParallelLayerGuidObliviousMachineMapping{{
          {
              BinaryTreePath{{}},
              post_mv1,
          },
      }};

      TensorSetMovement result = get_tensor_set_movement_across_split(
          pcg_get_transitive_reduction(pcg), split, pre_mapping, post_mapping);
      TensorSetMovement correct = TensorSetMovement{
          /*single_tensor_movements=*/{
              SingleTensorMovement{
                  /*parallel_tensor_shape=*/partitioned_input_shape,
                  /*src_machine_views=*/{pre_mv1},
                  /*dst_machine_views=*/{post_mv1},
              },
          },
      };

      CHECK(result == correct);
    }

    SUBCASE("does not include edges removed by transitive reduction") {}

    SUBCASE("single tensor, multiple consumers across split") {
      ParallelLayerAddedResult relu_3 =
          add_parallel_layer(pcg, relu_attrs, {get_only(relu_1.outputs)}, {});

      PCGBinarySeriesSplit split = PCGBinarySeriesSplit{
          make_pcg_series_split(
              make_pcg_series_split(
                  make_pcg_leaf_node(input.parallel_layer),
                  make_pcg_leaf_node(partition_input.parallel_layer)),
              make_pcg_leaf_node(relu_1.parallel_layer)),
          make_pcg_parallel_split(make_pcg_leaf_node(relu_2.parallel_layer),
                                  make_pcg_leaf_node(relu_3.parallel_layer)),
      };

      SUBCASE("consumers have same view") {
        auto pre_mapping = ParallelLayerGuidObliviousMachineMapping{{
            {
                BinaryTreePath{{
                    BinaryTreePathEntry::RIGHT_CHILD,
                }},
                pre_mv1,
            },
        }};

        auto post_mapping = ParallelLayerGuidObliviousMachineMapping{{
            {
                BinaryTreePath{{
                    BinaryTreePathEntry::LEFT_CHILD,
                }},
                post_mv1,
            },
            {
                BinaryTreePath{{
                    BinaryTreePathEntry::RIGHT_CHILD,
                }},
                post_mv1,
            },
        }};

        TensorSetMovement result = get_tensor_set_movement_across_split(
            pcg_get_transitive_reduction(pcg),
            split,
            pre_mapping,
            post_mapping);

        TensorSetMovement correct = TensorSetMovement{
            /*single_tensor_movements=*/{
                SingleTensorMovement{
                    /*parallel_tensor_shape=*/partitioned_input_shape,
                    /*src_machine_views=*/{pre_mv1},
                    /*dst_machine_views=*/{post_mv1},
                },
            },
        };

        CHECK(result == correct);
      }

      SUBCASE("consumers have different views") {
        auto pre_mapping = ParallelLayerGuidObliviousMachineMapping{{
            {
                BinaryTreePath{{
                    BinaryTreePathEntry::RIGHT_CHILD,
                }},
                pre_mv1,
            },
        }};

        auto post_mapping = ParallelLayerGuidObliviousMachineMapping{{
            {
                BinaryTreePath{{
                    BinaryTreePathEntry::LEFT_CHILD,
                }},
                post_mv1,
            },
            {
                BinaryTreePath{{
                    BinaryTreePathEntry::RIGHT_CHILD,
                }},
                post_mv2,
            },
        }};

        TensorSetMovement result = get_tensor_set_movement_across_split(
            pcg_get_transitive_reduction(pcg),
            split,
            pre_mapping,
            post_mapping);

        TensorSetMovement correct = TensorSetMovement{
            /*single_tensor_movements=*/{
                SingleTensorMovement{
                    /*parallel_tensor_shape=*/partitioned_input_shape,
                    /*src_machine_views=*/{pre_mv1},
                    /*dst_machine_views=*/{post_mv1, post_mv2},
                },
            },
        };

        CHECK(result == correct);
      }
    }

    SUBCASE("multiple tensors, multiple consumers across split") {
      ParallelLayerAddedResult relu_3 = add_parallel_layer(
          pcg, relu_attrs, {get_only(partition_input.outputs)}, {});

      ParallelLayerAddedResult relu_4 = add_parallel_layer(
          pcg,
          ew_add_attrs,
          {get_only(relu_1.outputs), get_only(relu_3.outputs)},
          {});

      PCGBinarySeriesSplit split = PCGBinarySeriesSplit{
          make_pcg_series_split(
              make_pcg_series_split(
                  make_pcg_leaf_node(input.parallel_layer),
                  make_pcg_leaf_node(partition_input.parallel_layer)),
              make_pcg_parallel_split(
                  make_pcg_leaf_node(relu_1.parallel_layer),
                  make_pcg_leaf_node(relu_3.parallel_layer))),
          make_pcg_parallel_split(make_pcg_leaf_node(relu_2.parallel_layer),
                                  make_pcg_leaf_node(relu_4.parallel_layer)),
      };

      auto pre_mapping = ParallelLayerGuidObliviousMachineMapping{{
          {
              BinaryTreePath{{
                  BinaryTreePathEntry::RIGHT_CHILD,
                  BinaryTreePathEntry::LEFT_CHILD,
              }},
              pre_mv1,
          },
          {
              BinaryTreePath{{
                  BinaryTreePathEntry::RIGHT_CHILD,
                  BinaryTreePathEntry::RIGHT_CHILD,
              }},
              pre_mv2,
          },
      }};

      auto post_mapping = ParallelLayerGuidObliviousMachineMapping{{
          {
              BinaryTreePath{{
                  BinaryTreePathEntry::LEFT_CHILD,
              }},
              post_mv1,
          },
          {
              BinaryTreePath{{
                  BinaryTreePathEntry::RIGHT_CHILD,
              }},
              post_mv2,
          },
      }};

      TensorSetMovement result = get_tensor_set_movement_across_split(
          pcg_get_transitive_reduction(pcg), split, pre_mapping, post_mapping);

      TensorSetMovement correct = TensorSetMovement{
          /*single_tensor_movements=*/{
              SingleTensorMovement{
                  /*parallel_tensor_shape=*/partitioned_input_shape,
                  /*src_machine_views=*/{pre_mv1},
                  /*dst_machine_views=*/{post_mv1, post_mv2},
              },
              SingleTensorMovement{
                  /*parallel_tensor_shape=*/partitioned_input_shape,
                  /*src_machine_views=*/{pre_mv2},
                  /*dst_machine_views=*/{post_mv2},
              },
          },
      };

      CHECK(result == correct);
    }
  }
}
