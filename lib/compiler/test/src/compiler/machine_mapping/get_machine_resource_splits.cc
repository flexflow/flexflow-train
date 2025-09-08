#include "compiler/machine_mapping/get_machine_resource_splits.h"
#include "pcg/machine_compute_specification.dtg.h"
#include "test/utils/doctest/fmt/pair.h"
#include "test/utils/doctest/fmt/unordered_set.h"
#include "utils/hash/pair.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("get_machine_resource_splits") {
    SUBCASE("returns no splits if no splits are possible") {
      MachineComputeResourceSlice input = MachineComputeResourceSlice{
        /*num_nodes=*/1_p,
        /*num_gpus_per_node=*/1_p,
      };

      std::unordered_set<std::pair<MachineComputeResourceSlice, MachineComputeResourceSlice>>
          result = get_machine_resource_splits(input);
      std::unordered_set<std::pair<MachineComputeResourceSlice, MachineComputeResourceSlice>>
          correct = {};

      CHECK(result == correct);
    }

    SUBCASE(
        "returns splits in gpu and node dimensions, but not at the same time") {
      MachineComputeResourceSlice input = MachineComputeResourceSlice{
        /*num_nodes=*/2_p,
        /*num_gpus_per_node=*/2_p,
      };

      std::unordered_set<std::pair<MachineComputeResourceSlice, MachineComputeResourceSlice>>
          result = get_machine_resource_splits(input);

      std::unordered_set<std::pair<MachineComputeResourceSlice, MachineComputeResourceSlice>>
          correct = {
              {
                MachineComputeResourceSlice{
                  /*num_nodes=*/2_p,
                  /*num_gpus_per_node=*/1_p,
                },
                MachineComputeResourceSlice{
                  /*num_nodes=*/2_p,
                  /*num_gpus_per_node=*/1_p,
                },
              },
              {
                MachineComputeResourceSlice{
                  /*num_nodes=*/1_p,
                  /*num_gpus_per_node=*/2_p,
                },
                MachineComputeResourceSlice{
                  /*num_nodes=*/1_p,
                  /*num_gpus_per_node=*/2_p,
                },
              },

          };

      CHECK(result == correct);
    }

    SUBCASE("returns splits in node dimension in powers of two") {
      SUBCASE("num_nodes is a power of 2") {
        MachineComputeResourceSlice input = MachineComputeResourceSlice{
          /*num_nodes=*/8_p,
          /*num_gpus_per_node=*/1_p,
        };

        std::unordered_set<
            std::pair<MachineComputeResourceSlice, MachineComputeResourceSlice>>
            result = get_machine_resource_splits(input);

        std::unordered_set<
            std::pair<MachineComputeResourceSlice, MachineComputeResourceSlice>>
            correct = {
                {
                  MachineComputeResourceSlice{
                    /*num_nodes=*/1_p,
                    /*num_gpus_per_node=*/1_p,
                  },
                  MachineComputeResourceSlice{
                    /*num_nodes=*/7_p,
                    /*num_gpus_per_node=*/1_p,
                  },
                },
                {
                  MachineComputeResourceSlice{
                    /*num_nodes=*/2_p,
                    /*num_gpus_per_node=*/1_p,
                  },
                  MachineComputeResourceSlice{
                    /*num_nodes=*/6_p,
                    /*num_gpus_per_node=*/1_p,
                  },
                },
                {
                  MachineComputeResourceSlice{
                    /*num_nodes=*/4_p,
                    /*num_gpus_per_node=*/1_p,
                  },
                  MachineComputeResourceSlice{
                    /*num_nodes=*/4_p,
                    /*num_gpus_per_node=*/1_p,
                  },
                },
                {
                  MachineComputeResourceSlice{
                    /*num_nodes=*/6_p,
                    /*num_gpus_per_node=*/1_p,
                  },
                  MachineComputeResourceSlice{
                    /*num_nodes=*/2_p,
                    /*num_gpus_per_node=*/1_p,
                  },
                },
                {
                  MachineComputeResourceSlice{
                    /*num_nodes=*/7_p,
                    /*num_gpus_per_node=*/1_p,
                  },
                  MachineComputeResourceSlice{
                    /*num_nodes=*/1_p,
                    /*num_gpus_per_node=*/1_p,
                  },
                },
            };

        CHECK(result == correct);
      }

      SUBCASE("num_nodes is not a power of 2") {
        MachineComputeResourceSlice input = MachineComputeResourceSlice{
          /*num_nodes=*/6_p,
          /*num_gpus_per_node=*/1_p,
        };

        std::unordered_set<
            std::pair<MachineComputeResourceSlice, MachineComputeResourceSlice>>
            result = get_machine_resource_splits(input);

        std::unordered_set<
            std::pair<MachineComputeResourceSlice, MachineComputeResourceSlice>>
            correct = {
                {
                  MachineComputeResourceSlice{
                    /*num_nodes=*/1_p,
                    /*num_gpus_per_node=*/1_p,
                  },
                  MachineComputeResourceSlice{
                    /*num_nodes=*/5_p,
                    /*num_gpus_per_node=*/1_p,
                  },
                },
                {
                  MachineComputeResourceSlice{
                    /*num_nodes=*/2_p,
                    /*num_gpus_per_node=*/1_p,
                  },
                  MachineComputeResourceSlice{
                    /*num_nodes=*/4_p,
                    /*num_gpus_per_node=*/1_p,
                  },
                },
                {
                  MachineComputeResourceSlice{
                    /*num_nodes=*/4_p,
                    /*num_gpus_per_node=*/1_p,
                  },
                  MachineComputeResourceSlice{
                    /*num_nodes=*/2_p,
                    /*num_gpus_per_node=*/1_p,
                  },
                },
                {
                  MachineComputeResourceSlice{
                    /*num_nodes=*/5_p,
                    /*num_gpus_per_node=*/1_p,
                  },
                  MachineComputeResourceSlice{
                    /*num_nodes=*/1_p,
                    /*num_gpus_per_node=*/1_p,
                  },
                },
            };

        CHECK(result == correct);
      }
    }

    SUBCASE("returns splits in gpu dimension in powers of two") {
      SUBCASE("num_gpus_per_node is a power of 2") {
        MachineComputeResourceSlice input = MachineComputeResourceSlice{
          /*num_nodes=*/1_p,
          /*num_gpus_per_node=*/8_p,
        };

        std::unordered_set<
            std::pair<MachineComputeResourceSlice, MachineComputeResourceSlice>>
            result = get_machine_resource_splits(input);

        std::unordered_set<
            std::pair<MachineComputeResourceSlice, MachineComputeResourceSlice>>
            correct = {
                {
                  MachineComputeResourceSlice{
                    /*num_nodes=*/1_p,
                    /*num_gpus_per_node=*/1_p,
                  },
                  MachineComputeResourceSlice{
                    /*num_nodes=*/1_p,
                    /*num_gpus_per_node=*/7_p,
                  },
                },
                {
                  MachineComputeResourceSlice{
                    /*num_nodes=*/1_p,
                    /*num_gpus_per_node=*/2_p,
                  },
                  MachineComputeResourceSlice{
                    /*num_nodes=*/1_p,
                    /*num_gpus_per_node=*/6_p,
                  },
                },
                {
                  MachineComputeResourceSlice{
                    /*num_nodes=*/1_p,
                    /*num_gpus_per_node=*/4_p,
                  },
                  MachineComputeResourceSlice{
                    /*num_nodes=*/1_p,
                    /*num_gpus_per_node=*/4_p,
                  },
                },
                {
                  MachineComputeResourceSlice{
                    /*num_nodes=*/1_p,
                    /*num_gpus_per_node=*/6_p,
                  },
                  MachineComputeResourceSlice{
                    /*num_nodes=*/1_p,
                    /*num_gpus_per_node=*/2_p,
                  },
                },
                {
                  MachineComputeResourceSlice{
                    /*num_nodes=*/1_p,
                    /*num_gpus_per_node=*/7_p,
                  },
                  MachineComputeResourceSlice{
                    /*num_nodes=*/1_p,
                    /*num_gpus_per_node=*/1_p,
                  },
                },
            };

        CHECK(result == correct);
      }

      SUBCASE("num_gpus_per_node is not a power of 2") {
        MachineComputeResourceSlice input = MachineComputeResourceSlice{
          /*num_nodes=*/1_p,
          /*num_gpus_per_node=*/6_p,
        };

        std::unordered_set<
            std::pair<MachineComputeResourceSlice, MachineComputeResourceSlice>>
            result = get_machine_resource_splits(input);

        std::unordered_set<
            std::pair<MachineComputeResourceSlice, MachineComputeResourceSlice>>
            correct = {
                {
                  MachineComputeResourceSlice{
                    /*num_nodes=*/1_p,
                    /*num_gpus_per_node=*/1_p,
                  },
                  MachineComputeResourceSlice{
                    /*num_nodes=*/1_p,
                    /*num_gpus_per_node=*/5_p,
                  },
                },
                {
                  MachineComputeResourceSlice{
                    /*num_nodes=*/1_p,
                    /*num_gpus_per_node=*/2_p,
                  },
                  MachineComputeResourceSlice{
                    /*num_nodes=*/1_p,
                    /*num_gpus_per_node=*/4_p,
                  },
                },
                {
                  MachineComputeResourceSlice{
                    /*num_nodes=*/1_p,
                    /*num_gpus_per_node=*/4_p,
                  },
                  MachineComputeResourceSlice{
                    /*num_nodes=*/1_p,
                    /*num_gpus_per_node=*/2_p,
                  },
                },
                {
                  MachineComputeResourceSlice{
                    /*num_nodes=*/1_p,
                    /*num_gpus_per_node=*/5_p,
                  },
                  MachineComputeResourceSlice{
                    /*num_nodes=*/1_p,
                    /*num_gpus_per_node=*/1_p,
                  },
                },
            };
      }
    }
  }
}
