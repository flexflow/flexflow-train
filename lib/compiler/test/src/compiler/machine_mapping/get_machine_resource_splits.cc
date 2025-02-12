#include "compiler/machine_mapping/get_machine_resource_splits.h"
#include "test/utils/doctest/fmt/pair.h"
#include "test/utils/doctest/fmt/unordered_set.h"
#include "utils/hash/pair.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("get_machine_resource_splits") {
    auto make_machine_spec = [](nonnegative_int num_nodes,
                                nonnegative_int num_gpus_per_node) {
      return MachineSpecification{
          /*num_nodes=*/num_nodes,
          /*num_cpus_per_node=*/1_n,
          /*num_gpus_per_node=*/num_gpus_per_node,
          /*inter_node_bandwidth=*/1.0,
          /*intra_node_bandwidth=*/1.0,
      };
    };

    SUBCASE("returns no splits if no splits are possible") {
      MachineSpecification input = make_machine_spec(/*num_nodes=*/1_n,
                                                     /*num_gpus_per_node=*/1_n);

      std::unordered_set<std::pair<MachineSpecification, MachineSpecification>>
          result = get_machine_resource_splits(input);
      std::unordered_set<std::pair<MachineSpecification, MachineSpecification>>
          correct = {};

      CHECK(result == correct);
    }

    SUBCASE(
        "returns splits in gpu and node dimensions, but not at the same time") {
      MachineSpecification input = make_machine_spec(/*num_nodes=*/2_n,
                                                     /*num_gpus_per_node=*/2_n);

      std::unordered_set<std::pair<MachineSpecification, MachineSpecification>>
          result = get_machine_resource_splits(input);

      std::unordered_set<std::pair<MachineSpecification, MachineSpecification>>
          correct = {
              {
                  make_machine_spec(/*num_nodes=*/2_n,
                                    /*num_gpus_per_node=*/1_n),
                  make_machine_spec(/*num_nodes=*/2_n,
                                    /*num_gpus_per_node=*/1_n),
              },
              {
                  make_machine_spec(/*num_nodes=*/1_n,
                                    /*num_gpus_per_node=*/2_n),
                  make_machine_spec(/*num_nodes=*/1_n,
                                    /*num_gpus_per_node=*/2_n),
              },

          };

      CHECK(result == correct);
    }

    SUBCASE("returns splits in node dimension in powers of two") {
      SUBCASE("num_nodes is a power of 2") {
        MachineSpecification input =
            make_machine_spec(/*num_nodes=*/8_n,
                              /*num_gpus_per_node=*/1_n);

        std::unordered_set<
            std::pair<MachineSpecification, MachineSpecification>>
            result = get_machine_resource_splits(input);

        std::unordered_set<
            std::pair<MachineSpecification, MachineSpecification>>
            correct = {
                {
                    make_machine_spec(/*num_nodes=*/1_n,
                                      /*num_gpus_per_node=*/1_n),
                    make_machine_spec(/*num_nodes=*/7_n,
                                      /*num_gpus_per_node=*/1_n),
                },
                {
                    make_machine_spec(/*num_nodes=*/2_n,
                                      /*num_gpus_per_node=*/1_n),
                    make_machine_spec(/*num_nodes=*/6_n,
                                      /*num_gpus_per_node=*/1_n),
                },
                {
                    make_machine_spec(/*num_nodes=*/4_n,
                                      /*num_gpus_per_node=*/1_n),
                    make_machine_spec(/*num_nodes=*/4_n,
                                      /*num_gpus_per_node=*/1_n),
                },
                {
                    make_machine_spec(/*num_nodes=*/6_n,
                                      /*num_gpus_per_node=*/1_n),
                    make_machine_spec(/*num_nodes=*/2_n,
                                      /*num_gpus_per_node=*/1_n),
                },
                {
                    make_machine_spec(/*num_nodes=*/7_n,
                                      /*num_gpus_per_node=*/1_n),
                    make_machine_spec(/*num_nodes=*/1_n,
                                      /*num_gpus_per_node=*/1_n),
                },
            };

        CHECK(result == correct);
      }

      SUBCASE("num_nodes is not a power of 2") {
        MachineSpecification input =
            make_machine_spec(/*num_nodes=*/6_n,
                              /*num_gpus_per_node=*/1_n);

        std::unordered_set<
            std::pair<MachineSpecification, MachineSpecification>>
            result = get_machine_resource_splits(input);

        std::unordered_set<
            std::pair<MachineSpecification, MachineSpecification>>
            correct = {
                {
                    make_machine_spec(/*num_nodes=*/1_n,
                                      /*num_gpus_per_node=*/1_n),
                    make_machine_spec(/*num_nodes=*/5_n,
                                      /*num_gpus_per_node=*/1_n),
                },
                {
                    make_machine_spec(/*num_nodes=*/2_n,
                                      /*num_gpus_per_node=*/1_n),
                    make_machine_spec(/*num_nodes=*/4_n,
                                      /*num_gpus_per_node=*/1_n),
                },
                {
                    make_machine_spec(/*num_nodes=*/4_n,
                                      /*num_gpus_per_node=*/1_n),
                    make_machine_spec(/*num_nodes=*/2_n,
                                      /*num_gpus_per_node=*/1_n),
                },
                {
                    make_machine_spec(/*num_nodes=*/5_n,
                                      /*num_gpus_per_node=*/1_n),
                    make_machine_spec(/*num_nodes=*/1_n,
                                      /*num_gpus_per_node=*/1_n),
                },
            };

        CHECK(result == correct);
      }
    }

    SUBCASE("returns splits in gpu dimension in powers of two") {
      SUBCASE("num_gpus_per_node is a power of 2") {
        MachineSpecification input =
            make_machine_spec(/*num_nodes=*/1_n,
                              /*num_gpus_per_node=*/8_n);

        std::unordered_set<
            std::pair<MachineSpecification, MachineSpecification>>
            result = get_machine_resource_splits(input);

        std::unordered_set<
            std::pair<MachineSpecification, MachineSpecification>>
            correct = {
                {
                    make_machine_spec(/*num_nodes=*/1_n,
                                      /*num_gpus_per_node=*/1_n),
                    make_machine_spec(/*num_nodes=*/1_n,
                                      /*num_gpus_per_node=*/7_n),
                },
                {
                    make_machine_spec(/*num_nodes=*/1_n,
                                      /*num_gpus_per_node=*/2_n),
                    make_machine_spec(/*num_nodes=*/1_n,
                                      /*num_gpus_per_node=*/6_n),
                },
                {
                    make_machine_spec(/*num_nodes=*/1_n,
                                      /*num_gpus_per_node=*/4_n),
                    make_machine_spec(/*num_nodes=*/1_n,
                                      /*num_gpus_per_node=*/4_n),
                },
                {
                    make_machine_spec(/*num_nodes=*/1_n,
                                      /*num_gpus_per_node=*/6_n),
                    make_machine_spec(/*num_nodes=*/1_n,
                                      /*num_gpus_per_node=*/2_n),
                },
                {
                    make_machine_spec(/*num_nodes=*/1_n,
                                      /*num_gpus_per_node=*/7_n),
                    make_machine_spec(/*num_nodes=*/1_n,
                                      /*num_gpus_per_node=*/1_n),
                },
            };

        CHECK(result == correct);
      }

      SUBCASE("num_gpus_per_node is not a power of 2") {
        MachineSpecification input =
            make_machine_spec(/*num_nodes=*/1_n,
                              /*num_gpus_per_node=*/6_n);

        std::unordered_set<
            std::pair<MachineSpecification, MachineSpecification>>
            result = get_machine_resource_splits(input);

        std::unordered_set<
            std::pair<MachineSpecification, MachineSpecification>>
            correct = {
                {
                    make_machine_spec(/*num_nodes=*/1_n,
                                      /*num_gpus_per_node=*/1_n),
                    make_machine_spec(/*num_nodes=*/1_n,
                                      /*num_gpus_per_node=*/5_n),
                },
                {
                    make_machine_spec(/*num_nodes=*/1_n,
                                      /*num_gpus_per_node=*/2_n),
                    make_machine_spec(/*num_nodes=*/1_n,
                                      /*num_gpus_per_node=*/4_n),
                },
                {
                    make_machine_spec(/*num_nodes=*/1_n,
                                      /*num_gpus_per_node=*/4_n),
                    make_machine_spec(/*num_nodes=*/1_n,
                                      /*num_gpus_per_node=*/2_n),
                },
                {
                    make_machine_spec(/*num_nodes=*/1_n,
                                      /*num_gpus_per_node=*/5_n),
                    make_machine_spec(/*num_nodes=*/1_n,
                                      /*num_gpus_per_node=*/1_n),
                },
            };
      }
    }
  }
}
