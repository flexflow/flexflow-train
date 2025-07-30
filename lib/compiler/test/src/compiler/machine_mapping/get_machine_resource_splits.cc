#include "compiler/machine_mapping/get_machine_resource_splits.h"
#include "test/utils/doctest/fmt/pair.h"
#include "test/utils/doctest/fmt/unordered_set.h"
#include "utils/hash/pair.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("get_machine_resource_splits") {
    auto make_machine_spec = [](positive_int num_nodes,
                                positive_int num_gpus_per_node) {
      return MachineComputeSpecification{
          /*num_nodes=*/num_nodes,
          /*num_cpus_per_node=*/1_p,
          /*num_gpus_per_node=*/num_gpus_per_node,
      };
    };

    SUBCASE("returns no splits if no splits are possible") {
      MachineComputeSpecification input = make_machine_spec(/*num_nodes=*/1_p,
                                                     /*num_gpus_per_node=*/1_p);

      std::unordered_set<std::pair<MachineComputeSpecification, MachineComputeSpecification>>
          result = get_machine_resource_splits(input);
      std::unordered_set<std::pair<MachineComputeSpecification, MachineComputeSpecification>>
          correct = {};

      CHECK(result == correct);
    }

    SUBCASE(
        "returns splits in gpu and node dimensions, but not at the same time") {
      MachineComputeSpecification input = make_machine_spec(/*num_nodes=*/2_p,
                                                     /*num_gpus_per_node=*/2_p);

      std::unordered_set<std::pair<MachineComputeSpecification, MachineComputeSpecification>>
          result = get_machine_resource_splits(input);

      std::unordered_set<std::pair<MachineComputeSpecification, MachineComputeSpecification>>
          correct = {
              {
                  make_machine_spec(/*num_nodes=*/2_p,
                                    /*num_gpus_per_node=*/1_p),
                  make_machine_spec(/*num_nodes=*/2_p,
                                    /*num_gpus_per_node=*/1_p),
              },
              {
                  make_machine_spec(/*num_nodes=*/1_p,
                                    /*num_gpus_per_node=*/2_p),
                  make_machine_spec(/*num_nodes=*/1_p,
                                    /*num_gpus_per_node=*/2_p),
              },

          };

      CHECK(result == correct);
    }

    SUBCASE("returns splits in node dimension in powers of two") {
      SUBCASE("num_nodes is a power of 2") {
        MachineComputeSpecification input =
            make_machine_spec(/*num_nodes=*/8_p,
                              /*num_gpus_per_node=*/1_p);

        std::unordered_set<
            std::pair<MachineComputeSpecification, MachineComputeSpecification>>
            result = get_machine_resource_splits(input);

        std::unordered_set<
            std::pair<MachineComputeSpecification, MachineComputeSpecification>>
            correct = {
                {
                    make_machine_spec(/*num_nodes=*/1_p,
                                      /*num_gpus_per_node=*/1_p),
                    make_machine_spec(/*num_nodes=*/7_p,
                                      /*num_gpus_per_node=*/1_p),
                },
                {
                    make_machine_spec(/*num_nodes=*/2_p,
                                      /*num_gpus_per_node=*/1_p),
                    make_machine_spec(/*num_nodes=*/6_p,
                                      /*num_gpus_per_node=*/1_p),
                },
                {
                    make_machine_spec(/*num_nodes=*/4_p,
                                      /*num_gpus_per_node=*/1_p),
                    make_machine_spec(/*num_nodes=*/4_p,
                                      /*num_gpus_per_node=*/1_p),
                },
                {
                    make_machine_spec(/*num_nodes=*/6_p,
                                      /*num_gpus_per_node=*/1_p),
                    make_machine_spec(/*num_nodes=*/2_p,
                                      /*num_gpus_per_node=*/1_p),
                },
                {
                    make_machine_spec(/*num_nodes=*/7_p,
                                      /*num_gpus_per_node=*/1_p),
                    make_machine_spec(/*num_nodes=*/1_p,
                                      /*num_gpus_per_node=*/1_p),
                },
            };

        CHECK(result == correct);
      }

      SUBCASE("num_nodes is not a power of 2") {
        MachineComputeSpecification input =
            make_machine_spec(/*num_nodes=*/6_p,
                              /*num_gpus_per_node=*/1_p);

        std::unordered_set<
            std::pair<MachineComputeSpecification, MachineComputeSpecification>>
            result = get_machine_resource_splits(input);

        std::unordered_set<
            std::pair<MachineComputeSpecification, MachineComputeSpecification>>
            correct = {
                {
                    make_machine_spec(/*num_nodes=*/1_p,
                                      /*num_gpus_per_node=*/1_p),
                    make_machine_spec(/*num_nodes=*/5_p,
                                      /*num_gpus_per_node=*/1_p),
                },
                {
                    make_machine_spec(/*num_nodes=*/2_p,
                                      /*num_gpus_per_node=*/1_p),
                    make_machine_spec(/*num_nodes=*/4_p,
                                      /*num_gpus_per_node=*/1_p),
                },
                {
                    make_machine_spec(/*num_nodes=*/4_p,
                                      /*num_gpus_per_node=*/1_p),
                    make_machine_spec(/*num_nodes=*/2_p,
                                      /*num_gpus_per_node=*/1_p),
                },
                {
                    make_machine_spec(/*num_nodes=*/5_p,
                                      /*num_gpus_per_node=*/1_p),
                    make_machine_spec(/*num_nodes=*/1_p,
                                      /*num_gpus_per_node=*/1_p),
                },
            };

        CHECK(result == correct);
      }
    }

    SUBCASE("returns splits in gpu dimension in powers of two") {
      SUBCASE("num_gpus_per_node is a power of 2") {
        MachineComputeSpecification input =
            make_machine_spec(/*num_nodes=*/1_p,
                              /*num_gpus_per_node=*/8_p);

        std::unordered_set<
            std::pair<MachineComputeSpecification, MachineComputeSpecification>>
            result = get_machine_resource_splits(input);

        std::unordered_set<
            std::pair<MachineComputeSpecification, MachineComputeSpecification>>
            correct = {
                {
                    make_machine_spec(/*num_nodes=*/1_p,
                                      /*num_gpus_per_node=*/1_p),
                    make_machine_spec(/*num_nodes=*/1_p,
                                      /*num_gpus_per_node=*/7_p),
                },
                {
                    make_machine_spec(/*num_nodes=*/1_p,
                                      /*num_gpus_per_node=*/2_p),
                    make_machine_spec(/*num_nodes=*/1_p,
                                      /*num_gpus_per_node=*/6_p),
                },
                {
                    make_machine_spec(/*num_nodes=*/1_p,
                                      /*num_gpus_per_node=*/4_p),
                    make_machine_spec(/*num_nodes=*/1_p,
                                      /*num_gpus_per_node=*/4_p),
                },
                {
                    make_machine_spec(/*num_nodes=*/1_p,
                                      /*num_gpus_per_node=*/6_p),
                    make_machine_spec(/*num_nodes=*/1_p,
                                      /*num_gpus_per_node=*/2_p),
                },
                {
                    make_machine_spec(/*num_nodes=*/1_p,
                                      /*num_gpus_per_node=*/7_p),
                    make_machine_spec(/*num_nodes=*/1_p,
                                      /*num_gpus_per_node=*/1_p),
                },
            };

        CHECK(result == correct);
      }

      SUBCASE("num_gpus_per_node is not a power of 2") {
        MachineComputeSpecification input =
            make_machine_spec(/*num_nodes=*/1_p,
                              /*num_gpus_per_node=*/6_p);

        std::unordered_set<
            std::pair<MachineComputeSpecification, MachineComputeSpecification>>
            result = get_machine_resource_splits(input);

        std::unordered_set<
            std::pair<MachineComputeSpecification, MachineComputeSpecification>>
            correct = {
                {
                    make_machine_spec(/*num_nodes=*/1_p,
                                      /*num_gpus_per_node=*/1_p),
                    make_machine_spec(/*num_nodes=*/1_p,
                                      /*num_gpus_per_node=*/5_p),
                },
                {
                    make_machine_spec(/*num_nodes=*/1_p,
                                      /*num_gpus_per_node=*/2_p),
                    make_machine_spec(/*num_nodes=*/1_p,
                                      /*num_gpus_per_node=*/4_p),
                },
                {
                    make_machine_spec(/*num_nodes=*/1_p,
                                      /*num_gpus_per_node=*/4_p),
                    make_machine_spec(/*num_nodes=*/1_p,
                                      /*num_gpus_per_node=*/2_p),
                },
                {
                    make_machine_spec(/*num_nodes=*/1_p,
                                      /*num_gpus_per_node=*/5_p),
                    make_machine_spec(/*num_nodes=*/1_p,
                                      /*num_gpus_per_node=*/1_p),
                },
            };
      }
    }
  }
}
