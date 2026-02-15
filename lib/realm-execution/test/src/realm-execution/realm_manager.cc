#include "realm-execution/realm_manager.h"
#include "internal/realm_test_utils.h"
#include "realm-execution/distributed_device_handle.h"
#include <doctest/doctest.h>

namespace test {

using namespace ::FlexFlow;
namespace Realm = ::FlexFlow::Realm;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("RealmManager") {
    std::vector<char *> fake_args =
        make_fake_realm_args(/*num_cpus=*/1_p, /*num_gpus=*/0_n);
    int fake_argc = fake_args.size();
    char **fake_argv = fake_args.data();

    // Initialize Realm
    RealmManager manager(&fake_argc, &fake_argv);

    // Launch a controller
    int some_data = 123;
    Realm::Event event = manager.start_controller([&](RealmContext &ctx) {
      // Data is captured and retains value
      ASSERT(some_data == 123);
    });
    // Need to block on the completion of the event to ensure we don't race,
    // because the lambda captures the environment
    event.wait();
  }
}

} // namespace test
