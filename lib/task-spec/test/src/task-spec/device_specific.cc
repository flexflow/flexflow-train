#include "task-spec/device_specific.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("DeviceSpecific") {
    DeviceSpecific<std::string> device_specific1 =
        DeviceSpecific<std::string>::create(
            device_id_t{MachineSpaceCoordinate{0_n, 0_n}, DeviceType::GPU},
            "hello world");

    DeviceSpecific<std::string> device_specific2 =
        DeviceSpecific<std::string>::create(
            device_id_t{MachineSpaceCoordinate{0_n, 1_n}, DeviceType::GPU},
            "hello world");

    std::string result1 = fmt::to_string(device_specific1);
    std::string result2 = fmt::to_string(device_specific2);

    CHECK(result1 != result2);
  }
}
