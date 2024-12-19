#include "compiler/machine_mapping/memory_optimization/machine_mapping_with_memory_result.h"
#include "pcg/machine_view.h"
#include <doctest/doctest.h>

using namespace FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("remove_non_pareto_optimal_machine_mapping_result") {
    MachineView machine_view_0 = MachineView{
        /*start=*/MachineSpaceCoordinate{
            /*node_idx=*/0,
            /*device_idx=*/0,
            /*device_type=*/DeviceType::GPU,
        },
        /*dimensions=*/
        {
            MachineViewDimension{
                stride_t{1},
                MachineSpecificationDimension::INTRA_NODE,
            },
        },
    };

    MachineView machine_view_1 = MachineView{
        /*start=*/MachineSpaceCoordinate{
            /*node_idx=*/0,
            /*device_idx=*/0,
            /*device_type=*/DeviceType::GPU,
        },
        /*dimensions=*/
        {
            MachineViewDimension{
                stride_t{2},
                MachineSpecificationDimension::INTRA_NODE,
            },
        },
    };

    MachineView machine_view_2 = MachineView{
        /*start=*/MachineSpaceCoordinate{
            /*node_idx=*/0,
            /*device_idx=*/0,
            /*device_type=*/DeviceType::GPU,
        },
        /*dimensions=*/
        {
            MachineViewDimension{
                stride_t{4},
                MachineSpecificationDimension::INTRA_NODE,
            },
        },
    };

    OpCostMetrics cost1 = OpCostMetrics{
        2.0,
        2,
    };
    OpCostMetrics cost2 = OpCostMetrics{
        4.0,
        1,
    };
    OpCostMetrics cost3 = OpCostMetrics{
        2.0,
        3,
    };

    SingleMachineMapping mm1 = SingleMachineMapping{
        cost1,
        ParallelLayerGuidObliviousMachineMapping{
            {
                {
                    BinaryTreePath{{}},
                    machine_view_0,
                },
            },
        },
    };

    SingleMachineMapping mm2 = SingleMachineMapping{
        cost2,
        ParallelLayerGuidObliviousMachineMapping{
            {
                {
                    BinaryTreePath{{}},
                    machine_view_1,
                },
            },
        },
    };

    SingleMachineMapping mm3 = SingleMachineMapping{
        cost3,
        ParallelLayerGuidObliviousMachineMapping{
            {
                {
                    BinaryTreePath{{}},
                    machine_view_2,
                },
            },
        },
    };

    SUBCASE("empty") {
      MachineMappingWithMemoryResult to_remove =
          empty_machine_mapping_with_memory_result();
      MachineMappingWithMemoryResult result =
          remove_non_pareto_optimal_machine_mapping_result(to_remove);
      MachineMappingWithMemoryResult correct =
          empty_machine_mapping_with_memory_result();

      CHECK(result == correct);
    }

    SUBCASE("no non-pareto_optimal") {
      MachineMappingWithMemoryResult to_remove = MachineMappingWithMemoryResult{
          {
              mm1,
              mm2,
          },
      };
      MachineMappingWithMemoryResult result =
          remove_non_pareto_optimal_machine_mapping_result(to_remove);
      MachineMappingWithMemoryResult correct = to_remove;

      CHECK(result == correct);
    }

    SUBCASE("non-pareto_optimal") {
      MachineMappingWithMemoryResult to_remove = MachineMappingWithMemoryResult{
          {
              mm1,
              mm2,
              mm3,
          },
      };
      MachineMappingWithMemoryResult result =
          remove_non_pareto_optimal_machine_mapping_result(to_remove);
      MachineMappingWithMemoryResult correct = MachineMappingWithMemoryResult{
          {
              mm1,
              mm2,
          },
      };

      CHECK(result == correct);
    }
  }

  TEST_CASE("series_combine(memory)") {
    MachineView machine_view_0 = MachineView{
        /*start=*/MachineSpaceCoordinate{
            /*node_idx=*/0,
            /*device_idx=*/0,
            /*device_type=*/DeviceType::GPU,
        },
        /*dimensions=*/
        {
            MachineViewDimension{
                stride_t{1},
                MachineSpecificationDimension::INTRA_NODE,
            },
        },
    };

    MachineView machine_view_1 = MachineView{
        /*start=*/MachineSpaceCoordinate{
            /*node_idx=*/0,
            /*device_idx=*/0,
            /*device_type=*/DeviceType::GPU,
        },
        /*dimensions=*/
        {
            MachineViewDimension{
                stride_t{2},
                MachineSpecificationDimension::INTRA_NODE,
            },
        },
    };

    OpCostMetrics pre_cost = OpCostMetrics{
        2.0,
        2,
    };
    MachineMappingWithMemoryResult pre = MachineMappingWithMemoryResult{{
        SingleMachineMapping{
            pre_cost,
            ParallelLayerGuidObliviousMachineMapping{
                {
                    {
                        BinaryTreePath{
                            {BinaryTreePathEntry::LEFT_CHILD},
                        },
                        machine_view_0,
                    },
                    {
                        BinaryTreePath{
                            {BinaryTreePathEntry::RIGHT_CHILD},
                        },
                        machine_view_1,
                    },
                },
            },
        },
    }};

    OpCostMetrics post_cost = OpCostMetrics{
        4.0,
        1,
    };

    MachineMappingWithMemoryResult post = MachineMappingWithMemoryResult{{
        SingleMachineMapping{
            post_cost,
            ParallelLayerGuidObliviousMachineMapping{
                {
                    {
                        BinaryTreePath{{}},
                        machine_view_1,
                    },
                },
            },
        },
    }};

    MachineMappingWithMemoryResult empty =
        empty_machine_mapping_with_memory_result();

    float comm_cost = 3.0;

    SUBCASE("pre is empty") {
      MachineMappingWithMemoryResult result = series_combine(
          comm_cost, empty, post, ParallelSplitTransformation::LthenR);
      MachineMappingWithMemoryResult correct = empty;

      CHECK(result == correct);
    }

    SUBCASE("post is empty") {
      MachineMappingWithMemoryResult result = series_combine(
          comm_cost, pre, empty, ParallelSplitTransformation::LthenR);
      MachineMappingWithMemoryResult correct = empty;

      CHECK(result == correct);
    }

    SUBCASE("both are nonempty") {
      MachineMappingWithMemoryResult no_parallel_split_transform =
          MachineMappingWithMemoryResult{
              {
                  SingleMachineMapping{
                      /*cost=*/OpCostMetrics{
                          pre_cost.runtime + comm_cost + post_cost.runtime,
                          pre_cost.memory + post_cost.memory,
                      },
                      /*machine_mapping=*/
                      ParallelLayerGuidObliviousMachineMapping{{
                          {
                              BinaryTreePath{{
                                  BinaryTreePathEntry::LEFT_CHILD,
                                  BinaryTreePathEntry::LEFT_CHILD,
                              }},
                              machine_view_0,
                          },
                          {
                              BinaryTreePath{{
                                  BinaryTreePathEntry::LEFT_CHILD,
                                  BinaryTreePathEntry::RIGHT_CHILD,
                              }},
                              machine_view_1,
                          },
                          {
                              BinaryTreePath{{
                                  BinaryTreePathEntry::RIGHT_CHILD,
                              }},
                              machine_view_1,
                          },
                      }},
                  },
              },
          };

      SUBCASE("parallel_split_transformation = std::nullopt") {
        MachineMappingWithMemoryResult result =
            series_combine(comm_cost, pre, post, std::nullopt);
        MachineMappingWithMemoryResult correct = no_parallel_split_transform;

        CHECK(result == correct);
      }

      SUBCASE("parallel_split_transformation = LthenR") {
        MachineMappingWithMemoryResult result = series_combine(
            comm_cost, pre, post, ParallelSplitTransformation::LthenR);
        MachineMappingWithMemoryResult correct = no_parallel_split_transform;

        CHECK(result == correct);
      }

      SUBCASE("parallel_split_transformation = RthenL") {
        MachineMappingWithMemoryResult result = series_combine(
            comm_cost, pre, post, ParallelSplitTransformation::RthenL);
        MachineMappingWithMemoryResult correct = MachineMappingWithMemoryResult{
            {
                SingleMachineMapping{
                    /*cost=*/OpCostMetrics{
                        pre_cost.runtime + comm_cost + post_cost.runtime,
                        pre_cost.memory + post_cost.memory,
                    },
                    /*machine_mapping=*/
                    ParallelLayerGuidObliviousMachineMapping{{
                        {
                            BinaryTreePath{{
                                BinaryTreePathEntry::RIGHT_CHILD,
                                BinaryTreePathEntry::LEFT_CHILD,
                            }},
                            machine_view_0,
                        },
                        {
                            BinaryTreePath{{
                                BinaryTreePathEntry::RIGHT_CHILD,
                                BinaryTreePathEntry::RIGHT_CHILD,
                            }},
                            machine_view_1,
                        },
                        {
                            BinaryTreePath{{
                                BinaryTreePathEntry::LEFT_CHILD,
                            }},
                            machine_view_1,
                        },
                    }},
                },
            },
        };

        CHECK(result == correct);
      }
    }
  }

  TEST_CASE("parallel_combine(memory)") {
    MachineView machine_view_0 = MachineView{
        /*start=*/MachineSpaceCoordinate{
            /*node_idx=*/0,
            /*device_idx=*/0,
            /*device_type=*/DeviceType::GPU,
        },
        /*dimensions=*/
        {
            MachineViewDimension{
                stride_t{1},
                MachineSpecificationDimension::INTRA_NODE,
            },
        },
    };

    MachineView machine_view_1 = MachineView{
        /*start=*/MachineSpaceCoordinate{
            /*node_idx=*/0,
            /*device_idx=*/0,
            /*device_type=*/DeviceType::GPU,
        },
        /*dimensions=*/
        {
            MachineViewDimension{
                stride_t{2},
                MachineSpecificationDimension::INTRA_NODE,
            },
        },
    };

    OpCostMetrics lhs_cost = OpCostMetrics{
        2.0,
        2,
    };
    MachineMappingWithMemoryResult lhs = MachineMappingWithMemoryResult{{
        SingleMachineMapping{
            lhs_cost,
            ParallelLayerGuidObliviousMachineMapping{
                {
                    {
                        BinaryTreePath{
                            {BinaryTreePathEntry::LEFT_CHILD},
                        },
                        machine_view_0,
                    },
                    {
                        BinaryTreePath{
                            {BinaryTreePathEntry::RIGHT_CHILD},
                        },
                        machine_view_1,
                    },
                },
            },
        },
    }};

    OpCostMetrics rhs_cost = OpCostMetrics{
        4.0,
        1,
    };
    MachineMappingWithMemoryResult rhs = MachineMappingWithMemoryResult{{
        SingleMachineMapping{
            rhs_cost,
            ParallelLayerGuidObliviousMachineMapping{
                {
                    {
                        BinaryTreePath{{}},
                        machine_view_1,
                    },
                },
            },
        },
    }};

    MachineMappingWithMemoryResult empty =
        empty_machine_mapping_with_memory_result();

    SUBCASE("lhs is empty") {
      MachineMappingWithMemoryResult result = parallel_combine(empty, rhs);
      MachineMappingWithMemoryResult correct = empty;

      CHECK(result == correct);
    }

    SUBCASE("rhs is empty") {
      MachineMappingWithMemoryResult result = parallel_combine(lhs, empty);
      MachineMappingWithMemoryResult correct = empty;

      CHECK(result == correct);
    }

    SUBCASE("both are nonempty") {
      MachineMappingWithMemoryResult result = parallel_combine(lhs, rhs);
      MachineMappingWithMemoryResult correct = MachineMappingWithMemoryResult{{
          SingleMachineMapping{
              /*cost=*/OpCostMetrics{
                  std::max(lhs_cost.runtime, rhs_cost.runtime),
                  std::max(lhs_cost.memory, rhs_cost.memory),
              },
              /*machine_mapping=*/
              ParallelLayerGuidObliviousMachineMapping{
                  {
                      {
                          BinaryTreePath{{BinaryTreePathEntry::LEFT_CHILD,
                                          BinaryTreePathEntry::LEFT_CHILD}},
                          machine_view_0,
                      },
                      {
                          BinaryTreePath{{BinaryTreePathEntry::LEFT_CHILD,
                                          BinaryTreePathEntry::RIGHT_CHILD}},
                          machine_view_1,
                      },
                      {
                          BinaryTreePath{{BinaryTreePathEntry::RIGHT_CHILD}},
                          machine_view_1,
                      },
                  },
              },
          },
      }};

      CHECK(result == correct);
    }
  }

  TEST_CASE("minimize_runtime(memory)") {
    MachineView machine_view_0 = MachineView{
        /*start=*/MachineSpaceCoordinate{
            /*node_idx=*/0,
            /*device_idx=*/0,
            /*device_type=*/DeviceType::GPU,
        },
        /*dimensions=*/
        {
            MachineViewDimension{
                stride_t{1},
                MachineSpecificationDimension::INTRA_NODE,
            },
        },
    };

    MachineView machine_view_1 = MachineView{
        /*start=*/MachineSpaceCoordinate{
            /*node_idx=*/0,
            /*device_idx=*/0,
            /*device_type=*/DeviceType::GPU,
        },
        /*dimensions=*/
        {
            MachineViewDimension{
                stride_t{2},
                MachineSpecificationDimension::INTRA_NODE,
            },
        },
    };

    MachineView machine_view_2 = MachineView{
        /*start=*/MachineSpaceCoordinate{
            /*node_idx=*/0,
            /*device_idx=*/0,
            /*device_type=*/DeviceType::GPU,
        },
        /*dimensions=*/
        {
            MachineViewDimension{
                stride_t{4},
                MachineSpecificationDimension::INTRA_NODE,
            },
        },
    };

    OpCostMetrics cost1 = OpCostMetrics{
        2.0,
        2,
    };
    OpCostMetrics cost2 = OpCostMetrics{
        4.0,
        1,
    };
    OpCostMetrics cost3 = OpCostMetrics{
        2.0,
        3,
    };

    SingleMachineMapping mm1 = SingleMachineMapping{
        cost1,
        ParallelLayerGuidObliviousMachineMapping{
            {
                {
                    BinaryTreePath{{}},
                    machine_view_0,
                },
            },
        },
    };

    SingleMachineMapping mm2 = SingleMachineMapping{
        cost2,
        ParallelLayerGuidObliviousMachineMapping{
            {
                {
                    BinaryTreePath{{}},
                    machine_view_1,
                },
            },
        },
    };

    SingleMachineMapping mm3 = SingleMachineMapping{
        cost3,
        ParallelLayerGuidObliviousMachineMapping{
            {
                {
                    BinaryTreePath{{}},
                    machine_view_2,
                },
            },
        },
    };

    MachineMappingWithMemoryResult result1 = MachineMappingWithMemoryResult{
        {
            mm1,
            mm2,
        },
    };

    MachineMappingWithMemoryResult result2 = MachineMappingWithMemoryResult{
        {
            mm2,
            mm3,
        },
    };

    MachineMappingWithMemoryResult result = minimize_runtime(result1, result2);
    MachineMappingWithMemoryResult correct = MachineMappingWithMemoryResult{
        {
            mm1,
            mm2,
        },
    };

    CHECK(result == correct);
  }
}
