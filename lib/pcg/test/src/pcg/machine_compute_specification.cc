#include "pcg/machine_compute_specification.h"
#include <doctest/doctest.h>

using namespace FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {

  TEST_CASE("MachineComputeSpecification") {
    MachineComputeSpecification ms = MachineComputeSpecification{
        /*num_nodes=*/4_p,
        /*num_cpus_per_node=*/16_p,
        /*num_gpus_per_node=*/8_p,
    };

    SUBCASE("get_num_gpus") {
      CHECK(get_num_gpus(ms) == 4 * 8);
    }

    SUBCASE("get_num_cpus") {
      CHECK(get_num_cpus(ms) == 4 * 16);
    }

    SUBCASE("get_num_devices") {
      CHECK(get_num_devices(ms, DeviceType::GPU) == 4 * 8);
      CHECK(get_num_devices(ms, DeviceType::CPU) == 16 * 4);
    }
  }
}
