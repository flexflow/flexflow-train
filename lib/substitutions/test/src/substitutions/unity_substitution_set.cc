#include "substitutions/unity_substitution_set.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("get_substitution_set") {
    MachineSpecification machine_spec = MachineSpecification{
        /*num_nodes=*/2_n,
        /*num_cpus_per_node=*/8_n,
        /*num_gpus_per_node=*/4_n,
        /*inter_node_bandwidth=*/0.0,
        /*intra_node_bandwidth=*/0.0,
    };

    std::vector<Substitution> result = get_substitution_set(machine_spec);

    CHECK(result.size() == 36);
  }
}
