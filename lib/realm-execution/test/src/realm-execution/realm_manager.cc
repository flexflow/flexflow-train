#include "realm-execution/realm_manager.h"
#include "realm-execution/parallel_computation_graph_instance/parallel_computation_graph_instance.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("RealmManager") {
    // Construct some fake command line for our test
    char fake_executable_name[] = "fake_executable_name";
    std::vector<char *> fake_args{fake_executable_name};
    int fake_argc = fake_args.size();
    char **fake_argv = fake_args.data();

    // Initialize Realm
    RealmManager manager(&fake_argc, &fake_argv);

    // Launch a controller
    int some_data = 123;
    FlexFlow::Realm::Event event = manager.start_controller(
        [&](RealmContext &ctx) { ASSERT(some_data == 123); });
    // Need to block on the completion of the event to ensure we don't race
    event.wait();
  }
}
