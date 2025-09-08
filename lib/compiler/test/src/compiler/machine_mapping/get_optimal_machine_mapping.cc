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

    MachineView mv1 = MachineView{
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

    MachineView mv2 = MachineView{
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

    MachineComputeResourceSlice full_machine_resources = MachineComputeResourceSlice{
        /*num_nodes=*/2_p,
        /*num_gpus_per_node=*/1_p,
    };

    MachineComputeResourceSlice split_machine_resources = MachineComputeResourceSlice{
        /*num_nodes=*/1_p,
        /*num_gpus_per_node=*/1_p,
    };

    auto allowed_machine_views1 =
        [&](UnmappedRuntimeOnlyOpCostEstimateKey const &,
            MachineComputeResourceSlice const &resources) {
          if (resources == full_machine_resources) {
            return std::unordered_set<MachineView>{mv1, mv2};
          } else {
            return std::unordered_set<MachineView>{mv2};
          }
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

    UnmappedRuntimeOnlyOpCostEstimateKey k1 =
        UnmappedRuntimeOnlyOpCostEstimateKey{
            /*op_attrs=*/PCGOperatorAttrs{InputAttrs{tensor_shape}},
            /*input_shapes=*/{},
            /*weight_shapes=*/{},
            /*output_shapes=*/{},
        };

    UnmappedRuntimeOnlyOpCostEstimateKey k2 =
        UnmappedRuntimeOnlyOpCostEstimateKey{
            /*op_attrs=*/PCGOperatorAttrs{ElementBinaryAttrs{
                /*type=*/OperatorType::EW_ADD,
                /*compute_type=*/DataType::FLOAT,
                /*should_broadcast_lhs=*/false,
                /*should_broadcast_rhs=*/false,
            }},
            /*input_shapes=*/{},
            /*weight_shapes=*/{},
            /*output_shapes=*/{},
        };

    ParallelTensorShape par_tensor_shape = lift_to_parallel(tensor_shape);

    TaskSpaceCoordinate empty_task_space_coord = TaskSpaceCoordinate{OrthotopeCoord{{}}};

    AbstractedDevice src_device = AbstractedDevice{
      /*operator_tree_path=*/binary_tree_root_path(),
      /*task_space_coordinate=*/empty_task_space_coord,
    };
    AbstractedDevice dst_device = src_device;

    AbstractedTensorSetMovement movement1 = AbstractedTensorSetMovement{{
      /*edge_to_size=*/{{
        AbstractedCommunicationEdge{
          /*src=*/src_device,
          /*dst=*/dst_device,
        },
        get_size_in_bytes(tensor_shape),
      }},
    }};

    ParallelLayerGuidObliviousMachineMapping mm1 =
        ParallelLayerGuidObliviousMachineMapping{{
            {binary_tree_root_path(), mv1},
        }};
    ParallelLayerGuidObliviousMachineMapping mm2 =
        ParallelLayerGuidObliviousMachineMapping{{
            {binary_tree_root_path(), mv2},
        }};

    auto map1 = std::unordered_map<RuntimeOnlyOpCostEstimateKey,
                                   RuntimeOnlyOpCostMetrics>{{
        {map_unmapped_runtime_only_op_cost_estimate_key(k1, mv1),
         RuntimeOnlyOpCostMetrics{/*forward_runtime=*/0.5_ms,
                                  /*backward_runtime=*/0.5_ms}},
        {map_unmapped_runtime_only_op_cost_estimate_key(k2, mv1),
         RuntimeOnlyOpCostMetrics{/*forward_runtime=*/1.0_ms,
                                  /*backward_runtime=*/1.0_ms}},
        {map_unmapped_runtime_only_op_cost_estimate_key(k1, mv2),
         RuntimeOnlyOpCostMetrics{/*forward_runtime=*/0.75_ms,
                                  /*backward_runtime=*/0.75_ms}},
        {map_unmapped_runtime_only_op_cost_estimate_key(k2, mv2),
         RuntimeOnlyOpCostMetrics{/*forward_runtime=*/1.25_ms,
                                  /*backward_runtime=*/1.25_ms}},
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

    RuntimeOnlyCostEstimator runtime_only_cost_estimator =
        make_fake_runtime_only_cost_estimator(
            map1,
            std::unordered_map<TensorSetMovement, milliseconds_t>{{
                {TensorSetMovement{{}}, 0.0_ms},
                {mk_tensor_set_movement(mv1, mv1),
                 0.1_ms},
                {mk_tensor_set_movement(mv2, mv2),
                 0.2_ms},
                {mk_tensor_set_movement(mv1, mv2),
                 0.3_ms},
                {mk_tensor_set_movement(mv2, mv1),
                 0.4_ms},
            }});

    MachineMappingContext context = MachineMappingContext{
      /*cost_estimator=*/runtime_only_cost_estimator,
      /*allowed_machine_views=*/allowed_machine_views1,
    };

    MachineMappingCache cache = empty_machine_mapping_cache();

    SUBCASE("single layer") {
      MachineMappingProblemTree problem_tree = make_leaf(k1);

      MachineMappingConstraints constraints =
          get_unconstrained_solution_for_layers(
              get_all_leaf_paths(problem_tree));

      MachineMappingResult result = get_optimal_machine_mapping(
          cache, context, problem_tree, full_machine_resources, constraints);
      MachineMappingResult correct = MachineMappingResult{
          FeasibleMachineMappingResult{
              /*runtime=*/1.0_ms,
              /*machine_mapping=*/
              ParallelLayerGuidObliviousMachineMapping{{
                  {binary_tree_root_path(), mv1},
              }},
          },
      };

      CHECK(result == correct);
    }

    SUBCASE("pair of layers in sequence") {
      MachineMappingProblemTree problem_tree =
          make_series_split(movement1, make_leaf(k1), make_leaf(k2));

      MachineMappingConstraints constraints =
          get_unconstrained_solution_for_layers(
              get_all_leaf_paths(problem_tree));

      MachineMappingResult result = get_optimal_machine_mapping(
          cache, context, problem_tree, full_machine_resources, constraints);
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
