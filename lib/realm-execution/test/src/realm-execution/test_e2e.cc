#include "realm-execution/parallel_computation_graph_instance/parallel_computation_graph_instance.h"
#include "realm-execution/realm_manager.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("RealmBackend e2e Training") {
    char fake_executable_name[] = "fake_executable_name";
    std::vector<char *> fake_args{fake_executable_name};
    int fake_argc = fake_args.size();
    char **fake_argv = fake_args.data();
    RealmManager manager(&fake_argc, &fake_argv);
    manager.start_controller([](RealmManager &manager) {});
  }
}
