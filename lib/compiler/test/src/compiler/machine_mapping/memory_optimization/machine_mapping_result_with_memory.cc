#include "compiler/machine_mapping/memory_optimization/machine_mapping_result_with_memory.h"
#include "pcg/machine_view.h"
#include <doctest/doctest.h>

using namespace FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("remove_non_dominating_machine_mapping_result") {
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

    CostMetric cost1 = CostMetric{
        2.0,
        2,
    };
    CostMetric cost2 = CostMetric{
        4.0,
        1,
    };
    CostMetric cost3 = CostMetric{
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
      MachineMappingResultWithMemory to_remove =
          empty_machine_mapping_result_with_memory();
      MachineMappingResultWithMemory result =
          remove_non_dominating_machine_mapping_result(to_remove);
      MachineMappingResultWithMemory correct =
          empty_machine_mapping_result_with_memory();

      CHECK(result == correct);
    }

    SUBCASE("no non-dominating") {
      MachineMappingResultWithMemory to_remove = MachineMappingResultWithMemory{
          {
              mm1,
              mm2,
          },
      };
      MachineMappingResultWithMemory result =
          remove_non_dominating_machine_mapping_result(to_remove);
      MachineMappingResultWithMemory correct = to_remove;

      CHECK(result == correct);
    }

    SUBCASE("non-dominating") {
      MachineMappingResultWithMemory to_remove = MachineMappingResultWithMemory{
          {
              mm1,
              mm2,
              mm3,
          },
      };
      MachineMappingResultWithMemory result =
          remove_non_dominating_machine_mapping_result(to_remove);
      MachineMappingResultWithMemory correct = MachineMappingResultWithMemory{
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

    CostMetric pre_cost = CostMetric{
        2.0,
        2,
    };
    MachineMappingResultWithMemory pre = MachineMappingResultWithMemory{{
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

    CostMetric post_cost = CostMetric{
        4.0,
        1,
    };

    MachineMappingResultWithMemory post = MachineMappingResultWithMemory{{
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

    MachineMappingResultWithMemory empty =
        empty_machine_mapping_result_with_memory();

    float comm_cost = 3.0;

    SUBCASE("pre is empty") {
      MachineMappingResultWithMemory result = series_combine(
          comm_cost, empty, post, ParallelSplitTransformation::LthenR);
      MachineMappingResultWithMemory correct = empty;

      CHECK(result == correct);
    }

    SUBCASE("post is empty") {
      MachineMappingResultWithMemory result = series_combine(
          comm_cost, pre, empty, ParallelSplitTransformation::LthenR);
      MachineMappingResultWithMemory correct = empty;

      CHECK(result == correct);
    }

    SUBCASE("both are nonempty") {
      MachineMappingResultWithMemory no_parallel_split_transform =
          MachineMappingResultWithMemory{
              {
                  SingleMachineMapping{
                      /*cost=*/CostMetric{
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
        MachineMappingResultWithMemory result =
            series_combine(comm_cost, pre, post, std::nullopt);
        MachineMappingResultWithMemory correct = no_parallel_split_transform;

        CHECK(result == correct);
      }

      SUBCASE("parallel_split_transformation = LthenR") {
        MachineMappingResultWithMemory result = series_combine(
            comm_cost, pre, post, ParallelSplitTransformation::LthenR);
        MachineMappingResultWithMemory correct = no_parallel_split_transform;

        CHECK(result == correct);
      }

      SUBCASE("parallel_split_transformation = RthenL") {
        MachineMappingResultWithMemory result = series_combine(
            comm_cost, pre, post, ParallelSplitTransformation::RthenL);
        MachineMappingResultWithMemory correct = MachineMappingResultWithMemory{
            {
                SingleMachineMapping{
                    /*cost=*/CostMetric{
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

    CostMetric lhs_cost = CostMetric{
        2.0,
        2,
    };
    MachineMappingResultWithMemory lhs = MachineMappingResultWithMemory{{
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

    CostMetric rhs_cost = CostMetric{
        4.0,
        1,
    };
    MachineMappingResultWithMemory rhs = MachineMappingResultWithMemory{{
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

    MachineMappingResultWithMemory empty =
        empty_machine_mapping_result_with_memory();

    SUBCASE("lhs is empty") {
      MachineMappingResultWithMemory result = parallel_combine(empty, rhs);
      MachineMappingResultWithMemory correct = empty;

      CHECK(result == correct);
    }

    SUBCASE("rhs is empty") {
      MachineMappingResultWithMemory result = parallel_combine(lhs, empty);
      MachineMappingResultWithMemory correct = empty;

      CHECK(result == correct);
    }

    SUBCASE("both are nonempty") {
      MachineMappingResultWithMemory result = parallel_combine(lhs, rhs);
      MachineMappingResultWithMemory correct = MachineMappingResultWithMemory{{
          SingleMachineMapping{
              /*cost=*/CostMetric{
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

    CostMetric cost1 = CostMetric{
        2.0,
        2,
    };
    CostMetric cost2 = CostMetric{
        4.0,
        1,
    };
    CostMetric cost3 = CostMetric{
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

    MachineMappingResultWithMemory result1 = MachineMappingResultWithMemory{
        {
            mm1,
            mm2,
        },
    };

    MachineMappingResultWithMemory result2 = MachineMappingResultWithMemory{
        {
            mm2,
            mm3,
        },
    };

    MachineMappingResultWithMemory result = minimize_runtime(result1, result2);
    MachineMappingResultWithMemory correct = MachineMappingResultWithMemory{
        {
            mm1,
            mm2,
        },
    };

    CHECK(result == correct);
  }
}
