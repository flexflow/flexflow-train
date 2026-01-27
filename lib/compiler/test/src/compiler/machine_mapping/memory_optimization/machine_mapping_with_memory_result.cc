#include "compiler/machine_mapping/memory_optimization/machine_mapping_with_memory_result.h"
#include "compiler/machine_mapping/machine_view.h"
#include "test/utils/doctest/fmt/unordered_set.h"
#include "test/utils/rapidcheck/some.h"
#include "utils/nonnegative_int/nonnegative_int.h"
#include <doctest/doctest.h>

using namespace FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("MachineMappingWithMemoryResult") {
    SUBCASE("initialization") {
      SUBCASE("throws if initialized with non-pareto-optimal elements") {
        CHECK_THROWS(MachineMappingWithMemoryResult{{
            ParetoOptimalMachineMapping{
                /*cost=*/OpCostMetrics{
                    /*forward_runtime=*/5_ms,
                    /*backward_runtime=*/5_ms,
                    /*memory_usage=*/6_bytes,
                },
                /*machine_mapping=*/
                some<ParallelLayerGuidObliviousMachineMapping>(),
            },
            ParetoOptimalMachineMapping{
                /*cost=*/OpCostMetrics{
                    /*forward_runtime=*/2_ms,
                    /*backward_runtime=*/4_ms,
                    /*memory_usage=*/5_bytes,
                },
                /*machine_mapping=*/
                some<ParallelLayerGuidObliviousMachineMapping>(),
            },
        }});
      }

      SUBCASE("allows elements with identical performance") {
        ParetoOptimalMachineMapping mapping1 = ParetoOptimalMachineMapping{
            /*cost=*/OpCostMetrics{
                /*forward_runtime=*/5_ms,
                /*backward_runtime=*/5_ms,
                /*memory_usage=*/6_bytes,
            },
            /*machine_mapping=*/
            some<ParallelLayerGuidObliviousMachineMapping>(),
        };

        ParetoOptimalMachineMapping mapping2 = ParetoOptimalMachineMapping{
            /*cost=*/OpCostMetrics{
                /*forward_runtime=*/5_ms,
                /*backward_runtime=*/5_ms,
                /*memory_usage=*/5_bytes,
            },
            /*machine_mapping=*/
            some<ParallelLayerGuidObliviousMachineMapping>(),
        };

        ParetoOptimalMachineMapping mapping3 = ParetoOptimalMachineMapping{
            /*cost=*/OpCostMetrics{
                /*forward_runtime=*/5_ms,
                /*backward_runtime=*/5_ms,
                /*memory_usage=*/6_bytes,
            },
            /*machine_mapping=*/
            some<ParallelLayerGuidObliviousMachineMapping>(),
        };

        MachineMappingWithMemoryResult mapping_result =
            MachineMappingWithMemoryResult{{
                mapping1,
                mapping2,
                mapping3,
            }};

        std::unordered_set<ParetoOptimalMachineMapping> result =
            mapping_result.get_pareto_frontier();

        std::unordered_set<ParetoOptimalMachineMapping> correct = {
            mapping1,
            mapping2,
            mapping3,
        };

        CHECK(result == correct);
      }

      SUBCASE("allows empty set") {
        MachineMappingWithMemoryResult mapping_result =
            MachineMappingWithMemoryResult{{}};

        std::unordered_set<ParetoOptimalMachineMapping> result =
            mapping_result.get_pareto_frontier();

        std::unordered_set<ParetoOptimalMachineMapping> correct = {};

        CHECK(result == correct);
      }
    }
  }

  TEST_CASE("series_combine(float, MachineMappingWithMemoryResult const &, "
            "MachineMappingWithMemoryResult const &, "
            "std::optional<ParallelSplitTransformation> const&)") {
    MachineView machine_view_0 = MachineView{
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

    MachineView machine_view_1 = MachineView{
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

    OpCostMetrics pre_cost = OpCostMetrics{
        /*forward_runtime=*/2_ms,
        /*backward_runtime=*/2_ms,
        /*memory_usage=*/2_bytes,
    };
    MachineMappingWithMemoryResult pre = MachineMappingWithMemoryResult{{
        ParetoOptimalMachineMapping{
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
        /*forward_runtime=*/4_ms,
        /*backward_runtime=*/4_ms,
        /*memory_usage=*/1_bytes,
    };

    MachineMappingWithMemoryResult post = MachineMappingWithMemoryResult{{
        ParetoOptimalMachineMapping{
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

    milliseconds_t comm_cost = 3_ms;

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
                  ParetoOptimalMachineMapping{
                      /*cost=*/OpCostMetrics{
                          /*forward_runtime=*/pre_cost.forward_runtime +
                              comm_cost + post_cost.forward_runtime,
                          /*backward_runtime=*/pre_cost.backward_runtime +
                              comm_cost + post_cost.backward_runtime,
                          /*memory_usage=*/pre_cost.memory_usage +
                              post_cost.memory_usage,
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
                ParetoOptimalMachineMapping{
                    /*cost=*/OpCostMetrics{
                        /*forward_runtime=*/pre_cost.forward_runtime +
                            comm_cost + post_cost.forward_runtime,
                        /*backward_runtime=*/pre_cost.backward_runtime +
                            comm_cost + post_cost.backward_runtime,
                        /*memory_usage=*/pre_cost.memory_usage +
                            post_cost.memory_usage,
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

  TEST_CASE("parallel_combine(float, MachineMappingWithMemoryResult const &, "
            "MachineMappingWithMemoryResult const &, "
            "std::optional<ParallelSplitTransformation> const&)") {
    MachineView machine_view_0 = MachineView{
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

    MachineView machine_view_1 = MachineView{
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

    OpCostMetrics lhs_cost = OpCostMetrics{
        /*forward_runtime=*/2_ms,
        /*backward_runtime=*/2_ms,
        /*memory_usage=*/2_bytes,
    };
    MachineMappingWithMemoryResult lhs = MachineMappingWithMemoryResult{{
        ParetoOptimalMachineMapping{
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
        /*forward_runtime=*/4_ms,
        /*backward_runtime=*/4_ms,
        /*memory_usage=*/1_bytes,
    };
    MachineMappingWithMemoryResult rhs = MachineMappingWithMemoryResult{{
        ParetoOptimalMachineMapping{
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

    MachineResourceSplit split = MachineResourceSplit{
        /*offset=*/3_p,
        /*dimension=*/MachineSpecificationDimension::INTER_NODE,
    };

    SUBCASE("lhs is empty") {
      MachineMappingWithMemoryResult result =
          parallel_combine(split, empty, rhs);
      MachineMappingWithMemoryResult correct = empty;

      CHECK(result == correct);
    }

    SUBCASE("rhs is empty") {
      MachineMappingWithMemoryResult result =
          parallel_combine(split, lhs, empty);
      MachineMappingWithMemoryResult correct = empty;

      CHECK(result == correct);
    }

    SUBCASE("both are nonempty") {
      MachineMappingWithMemoryResult result = parallel_combine(split, lhs, rhs);

      MachineView translated_machine_view_1 = MachineView{
          /*start=*/MachineSpaceCoordinate{
              /*node_idx=*/3_n,
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

      MachineMappingWithMemoryResult correct = MachineMappingWithMemoryResult{{
          ParetoOptimalMachineMapping{
              /*cost=*/OpCostMetrics{
                  /*forward_runtime=*/std::max(lhs_cost.forward_runtime,
                                               rhs_cost.forward_runtime),
                  /*backward_runtime=*/
                  std::max(lhs_cost.backward_runtime,
                           rhs_cost.backward_runtime),
                  /*memory_usage=*/
                  std::max(lhs_cost.memory_usage, rhs_cost.memory_usage),
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
                          translated_machine_view_1,
                      },
                  },
              },
          },
      }};

      CHECK(result == correct);
    }
  }

  TEST_CASE("minimize_runtime(MachineMappingWithMemoryResult, "
            "MachineMappingWithMemoryResult)") {
    MachineView machine_view_0 = MachineView{
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

    MachineView machine_view_1 = MachineView{
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

    MachineView machine_view_2 = MachineView{
        /*start=*/MachineSpaceCoordinate{
            /*node_idx=*/0_n,
            /*device_idx=*/0_n,
            /*device_type=*/DeviceType::GPU,
        },
        /*dimensions=*/
        {
            MachineViewDimension{
                stride_t{4_p},
                MachineSpecificationDimension::INTRA_NODE,
            },
        },
    };

    OpCostMetrics cost1 = OpCostMetrics{
        /*forward_runtime=*/2_ms,
        /*backward_runtime=*/2_ms,
        /*memory_usage=*/2_bytes,
    };
    OpCostMetrics cost2 = OpCostMetrics{
        /*forward_runtime=*/4_ms,
        /*backward_runtime=*/4_ms,
        /*memory_usage=*/1_bytes,
    };
    OpCostMetrics cost3 = OpCostMetrics{
        /*forward_runtime=*/2.5_ms,
        /*backward_runtime=*/2.5_ms,
        /*memory_usage=*/3_bytes,
    };

    ParetoOptimalMachineMapping mm1 = ParetoOptimalMachineMapping{
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

    ParetoOptimalMachineMapping mm2 = ParetoOptimalMachineMapping{
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

    ParetoOptimalMachineMapping mm3 = ParetoOptimalMachineMapping{
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

    MachineMappingWithMemoryResult mapping_result1 =
        MachineMappingWithMemoryResult{
            {
                mm1,
                mm2,
            },
        };

    MachineMappingWithMemoryResult mapping_result2 =
        MachineMappingWithMemoryResult{
            {
                mm2,
                mm3,
            },
        };

    MachineMappingWithMemoryResult result =
        minimize_runtime(mapping_result1, mapping_result2);
    MachineMappingWithMemoryResult correct = MachineMappingWithMemoryResult{
        {
            mm1,
            mm2,
        },
    };

    CHECK(result == correct);
  }
}
