#include "task-spec/device_specific.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("DeviceSpecific") {
    DeviceSpecific<std::string> device_specific1 =
        DeviceSpecific<std::string>::create(device_id_t{gpu_id_t{0_n}},
                                            "hello world");

    DeviceSpecific<std::string> device_specific2 =
        DeviceSpecific<std::string>::create(device_id_t{gpu_id_t{1_n}},
                                            "hello world");

    std::string result1 = fmt::to_string(device_specific1);
    std::string result2 = fmt::to_string(device_specific2);

    CHECK(result1 != result2);
  }
}
