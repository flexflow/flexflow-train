#include "task-spec/device_specific.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("DeviceSpecific") {
    DeviceSpecific<std::string> device_specific =
        DeviceSpecific<std::string>::create(device_id_t{gpu_id_t{1_n}},
                                            "hello world");

    std::string result = fmt::to_string(device_specific);
    std::string correct = "hi";

    ASSERT(result == correct);
  }
}
