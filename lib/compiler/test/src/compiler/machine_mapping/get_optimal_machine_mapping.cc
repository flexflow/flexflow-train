#include "compiler/machine_mapping/get_optimal_machine_mapping.h"
#include "compiler/cost_estimator/runtime_only_op_cost_estimate_key.dtg.h"
#include "compiler/cost_estimator/runtime_only_op_cost_metrics.dtg.h"
#include "compiler/machine_mapping/abstracted_tensor_set_movement/abstracted_tensor_set_movement.h"
#include "compiler/machine_mapping/machine_compute_resource_slice.h"
#include "compiler/machine_mapping/machine_mapping_cache.h"
#include "compiler/machine_mapping/machine_mapping_constraints.h"
#include "compiler/machine_mapping/machine_mapping_problem_tree/machine_mapping_problem_tree.h"
#include "compiler/machine_mapping/machine_mapping_problem_tree/unmapped_op_cost_estimate_key.h"
#include "compiler/machine_mapping/machine_mapping_problem_tree/unmapped_runtime_only_op_cost_estimate_key.h"
#include "internal/runtime_only_cost_estimator_for_test.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "pcg/machine_view.h"
#include "pcg/parallel_computation_graph/parallel_computation_graph_builder.h"
#include "utils/containers/get_only.h"
#include "utils/full_binary_tree/binary_tree_path.h"
#include "utils/nonnegative_int/nonnegative_int.h"
#include <doctest/doctest.h>

using namespace FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("get_optimal_machine_mapping") {
    auto make_leaf = [](UnmappedRuntimeOnlyOpCostEstimateKey const &k) {
      return MachineMappingProblemTree{k};
    };

    auto make_series_split =
        [](AbstractedTensorSetMovement const &tensor_set_movement,
           MachineMappingProblemTree const &lhs,
           MachineMappingProblemTree const &rhs) {
          return MachineMappingProblemTree{
              MMProblemTreeSeriesSplit{
                  /*tensor_set_movement=*/tensor_set_movement,
                  /*left_child=*/lhs,
                  /*right_child=*/rhs,
              },
          };
        };

    auto make_parallel_split = [](MachineMappingProblemTree const &lhs,
                                  MachineMappingProblemTree const &rhs) {
      return MachineMappingProblemTree{
          MMProblemTreeParallelSplit{
              /*left_child=*/lhs,
              /*right_child=*/rhs,
          },
      };
    };

    MachineView mv_stride_1 = MachineView{
        /*start=*/MachineSpaceCoordinate{
            /*node_idx=*/0_n,
            /*device_idx=*/0_n,
            /*device_type=*/DeviceType::GPU,
        },
        /*dimensions=*/
        {
            MachineViewDimension{
                stride_t{1_p},
                MachineSpecificationDimension::INTRA_NODE,
            },
        },
    };

    MachineView mv_stride_2 = MachineView{
        /*start=*/MachineSpaceCoordinate{
            /*node_idx=*/0_n,
            /*device_idx=*/0_n,
            /*device_type=*/DeviceType::GPU,
        },
        /*dimensions=*/
        {
            MachineViewDimension{
                stride_t{2_p},
                MachineSpecificationDimension::INTRA_NODE,
            },
        },
    };

    MachineComputeResourceSlice four_nodes_resources = MachineComputeResourceSlice{
        /*num_nodes=*/4_p,
        /*num_gpus_per_node=*/1_p,
    };

    MachineComputeResourceSlice three_nodes_resources = MachineComputeResourceSlice{
      /*num_nodes=*/3_p,
      /*num_gpus_per_node=*/1_p,
    };

    MachineComputeResourceSlice two_nodes_resources = MachineComputeResourceSlice{
      /*num_nodes=*/2_p,
      /*num_gpus_per_node=*/1_p,
    };

    MachineComputeResourceSlice one_node_resources = MachineComputeResourceSlice{
        /*num_nodes=*/1_p,
        /*num_gpus_per_node=*/1_p,
    };

    TensorShape tensor_shape = TensorShape{
        TensorDims{
            FFOrdered{
                10_p,
                8_p,
            },
        },
        DataType::FLOAT,
    };

    TaskSpaceCoordinate empty_task_space_coord = TaskSpaceCoordinate{OrthotopeCoord{{}}};

    BinaryTreePath src_path = binary_tree_root_path();
    TaskSpaceCoordinate src_coord = empty_task_space_coord;

    AbstractedDevice dst_device = AbstractedDevice{
      /*operator_tree_path=*/binary_tree_root_path(),
      /*task_space_coordinate=*/empty_task_space_coord,
    };

    AbstractedTensorSetMovement movement1 = AbstractedTensorSetMovement{
      /*single_tensor_movements=*/{
        AbstractedSingleTensorMovement{
          /*src_op_tree_path=*/src_path,
          /*edge_to_size=*/{
            {
              AbstractedSingleTensorCommunicationEdge{
                /*src_coord=*/src_coord,
                /*dst=*/dst_device,
              },
              get_size_in_bytes(tensor_shape),
            },
          },
        },
      },
    };

    ParallelLayerGuidObliviousMachineMapping mm1 =
        ParallelLayerGuidObliviousMachineMapping{{
            {binary_tree_root_path(), mv1},
        }};
    ParallelLayerGuidObliviousMachineMapping mm2 =
        ParallelLayerGuidObliviousMachineMapping{{
            {binary_tree_root_path(), mv2},
        }};

    OperatorTaskSpace trivial_task_space = OperatorTaskSpace{MinimalOrthotope{{}}};

    auto mk_tensor_set_movement = [&](
      MachineView const &src_mv, 
      MachineView const &dst_mv) {

      MachineSpaceStencil src_stencil = MachineSpaceStencil{
        /*operator_task_space=*/trivial_task_space,
        /*machine_view=*/src_mv,
      };

      MachineSpaceStencil dst_stencil = MachineSpaceStencil{
        /*operator_task_space=*/trivial_task_space,
        /*machine_view=*/dst_mv,
      };

      return concretize_abstracted_tensor_set_movement(
        movement1,
        /*pre_machine_stencils=*/{{binary_tree_root_path(), src_stencil}},
        /*post_machine_stencils=*/{{binary_tree_root_path(), dst_stencil}});
    };

    MachineMappingCache cache = empty_machine_mapping_cache();

    ParallelTensorShape par_tensor_shape =
      lift_to_parallel_with_degrees(
        tensor_shape,
        ParallelTensorDimDegrees{
          /*sum_degree=*/SumDegree{1_p},
          /*discard_copy_degree=*/DiscardCopyDegree{1_p},
          /*shard_degrees=*/FFOrdered<positive_int>{
            2_p,
            1_p,
          },
        });

    UnmappedRuntimeOnlyOpCostEstimateKey k1 = UnmappedRuntimeOnlyOpCostEstimateKey{
        /*op_attrs=*/PCGOperatorAttrs{ElementUnaryAttrs{
            /*type=*/OperatorType::GELU,
            /*scalar=*/std::nullopt,
        }},
      /*input_shapes=*/{par_tensor_shape},
      /*weight_shapes=*/{},
      /*output_shapes=*/{par_tensor_shape},
    };

    UnmappedRuntimeOnlyOpCostEstimateKey k2 = UnmappedRuntimeOnlyOpCostEstimateKey{
        /*op_attrs=*/PCGOperatorAttrs{ElementUnaryAttrs{
            /*type=*/OperatorType::RELU,
            /*scalar=*/std::nullopt,
        }},
        /*input_shapes=*/{par_tensor_shape},
        /*weight_shapes=*/{},
        /*output_shapes=*/{par_tensor_shape},
    };


    auto mk_cost_metrics = [&](float cost) {
      return  RuntimeOnlyOpCostMetrics{
        /*forward_runtime=*/milliseconds_t{cost},
        /*backward_runtime=*/milliseconds_t{cost},
      };
    };

    SUBCASE("single layer") {
      MachineMappingProblemTree problem_tree = make_leaf(k1);

      MachineMappingConstraints constraints =
          get_unconstrained_solution_for_layers(
              get_all_leaf_paths(problem_tree));

      auto allowed_machine_views =
          [&](UnmappedRuntimeOnlyOpCostEstimateKey const &k,
              MachineComputeResourceSlice const &resources) {
            ASSERT(k == k1);
            ASSERT(resources == four_nodes_resources);
            return std::unordered_set<MachineView>{
              mv_stride_1,
              mv_stride_2,
            };
          };

      RuntimeOnlyCostEstimator runtime_only_cost_estimator =
          make_fake_runtime_only_cost_estimator(
              {
                {
                  map_unmapped_runtime_only_op_cost_estimate_key(k2, mv_stride_1),
                  mk_cost_metrics(0.5),
                },
                {
                  map_unmapped_runtime_only_op_cost_estimate_key(k2, mv_stride_2),
                  mk_cost_metrics(1),
                },
              },
              std::unordered_map<TensorSetMovement, milliseconds_t>{{}});

      MachineMappingContext context = MachineMappingContext{
        /*cost_estimator=*/runtime_only_cost_estimator,
        /*allowed_machine_views=*/allowed_machine_views,
      };

      MachineMappingResult result = get_optimal_machine_mapping(
          cache, context, problem_tree, four_nodes_resources, constraints);

      MachineMappingResult correct = MachineMappingResult{
          FeasibleMachineMappingResult{
              /*runtime=*/2_ms,
              /*machine_mapping=*/
              ParallelLayerGuidObliviousMachineMapping{{
                  {binary_tree_root_path(), mv_stride_1},
              }},
          },
      };

      CHECK(result == correct);
    }

    SUBCASE("pair of layers in sequence") {
      MachineMappingProblemTree problem_tree =
          make_series_split(movement1, make_leaf(k1), make_leaf(k2));

      RuntimeOnlyCostEstimator runtime_only_cost_estimator =
          make_fake_runtime_only_cost_estimator(
              std::unordered_map<RuntimeOnlyOpCostEstimateKey, RuntimeOnlyOpCostMetrics>{{
                {
                  map_unmapped_runtime_only_op_cost_estimate_key(k2, mv_stride_1),
                  mk_cost_metrics(0.5),
                },
                {
                  map_unmapped_runtime_only_op_cost_estimate_key(k2, mv_stride_2),
                  mk_cost_metrics(0.5),
                },
                {
                  map_unmapped_runtime_only_op_cost_estimate_key(k3, mv_stride_1),
                  mk_cost_metrics(0.5),
                },
                {
                  map_unmapped_runtime_only_op_cost_estimate_key(k3, mv_stride_2),
                  mk_cost_metrics(0.5),
                },
              }},
              std::unordered_map<TensorSetMovement, milliseconds_t>{{
                  {
                    TensorSetMovement{{}}, 
                    0.0_ms,
                  },
                  {
                    mk_tensor_set_movement(mv_stride_1, mv_stride_2),
                    0.1_ms,
                  },
                  {
                    mk_tensor_set_movement(mv2, mv2),
                    0.2_ms,
                  },
                  {
                    mk_tensor_set_movement(mv1, mv2),
                    0.3_ms,
                  },
                  {
                    mk_tensor_set_movement(mv2, mv1),
                    0.4_ms,
                  },
              }});

      auto allowed_machine_views =
        [&](UnmappedRuntimeOnlyOpCostEstimateKey const &k,
            MachineComputeResourceSlice const &resources) {
          if (resources == four_nodes_resources) {
            return std::unordered_set<MachineView>{mv2, mv3};
          } else if (resources == three_nodes_resources) {
            return std::unordered_set<MachineView>{mv2, mv3};
          } else if (resources == two_nodes_resources) {
            return std::unordered_set<MachineView>{mv2};
          } else {
            return std::unordered_set<MachineView>{};
          }
        };

      MachineMappingConstraints constraints =
          get_unconstrained_solution_for_layers(
              get_all_leaf_paths(problem_tree));

      MachineMappingContext context = MachineMappingContext{
        /*cost_estimator=*/runtime_only_cost_estimator,
        /*allowed_machine_views=*/allowed_machine_views,
      };

      MachineMappingResult result = get_optimal_machine_mapping(
          cache, context, problem_tree, four_nodes_resources, constraints);

      MachineMappingResult correct = MachineMappingResult{
          FeasibleMachineMappingResult{
              /*runtime=*/1.0_ms + 2.0_ms + 0.1_ms,
              /*machine_mapping=*/
              ParallelLayerGuidObliviousMachineMapping{{
                  {
                      BinaryTreePath{{
                          BinaryTreePathEntry::LEFT_CHILD,
                      }},
                      mv1,
                  },
                  {
                      BinaryTreePath{{
                          BinaryTreePathEntry::RIGHT_CHILD,
                      }},
                      mv1,
                  },
              }},
          },
      };

      CHECK(result == correct);
    }

    SUBCASE("pair of layers in parallel") {
      MachineMappingProblemTree problem_tree =
          make_parallel_split(make_leaf(k1), make_leaf(k2));

      MachineMappingConstraints constraints =
          get_unconstrained_solution_for_layers(
              get_all_leaf_paths(problem_tree));

      MachineMappingResult result = get_optimal_machine_mapping(
          cache, context, problem_tree, full_machine_resources, constraints);
      MachineMappingResult correct = MachineMappingResult{
          FeasibleMachineMappingResult{
              /*runtime=*/2.5_ms,
              /*machine_mapping=*/
              ParallelLayerGuidObliviousMachineMapping{{
                  {
                      BinaryTreePath{{
                          BinaryTreePathEntry::LEFT_CHILD,
                      }},
                      mv2,
                  },
                  {
                      BinaryTreePath{{
                          BinaryTreePathEntry::RIGHT_CHILD,
                      }},
                      mv2,
                  },
              }},
          },
      };

      CHECK(result == correct);
    }
  }
}
