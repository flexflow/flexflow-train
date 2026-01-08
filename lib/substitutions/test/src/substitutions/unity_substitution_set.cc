#include "substitutions/unity_substitution_set.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("get_substitution_set") {
    MachineComputeSpecification machine_spec = MachineComputeSpecification{
        /*num_nodes=*/2_p,
        /*num_cpus_per_node=*/8_p,
        /*num_gpus_per_node=*/4_p,
    };

    std::vector<Substitution> result = get_substitution_set(machine_spec);

    CHECK(result.size() == 36);
  }
}
