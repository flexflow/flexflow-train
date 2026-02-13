#include "realm-execution/distributed_device_handle.h"
#include "realm-execution/realm_manager.h"
#include <doctest/doctest.h>

namespace test {

using namespace ::FlexFlow;
namespace Realm = ::FlexFlow::Realm;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("DistributedDeviceHandle") {
    // Construct some fake command line for our test
    char fake_executable_name[] = "fake_executable_name";
    char arg0[] = "-ll:cpu";
    char arg1[] = "2";
    std::vector<char *> fake_args{fake_executable_name, arg0, arg1};
    int fake_argc = fake_args.size();
    char **fake_argv = fake_args.data();

    RealmManager manager(&fake_argc, &fake_argv);

    (void)manager.start_controller([](RealmContext &ctx) {
      DistributedDeviceHandle handle = create_distributed_device_handle(
          /*ctx=*/ctx,
          /*workSpaceSize=*/1024 * 1024,
          /*allowTensorOpMathConversion=*/true);

      // Make sure we have handles for the processors we're expecting
      Realm::Machine::ProcessorQuery pq(Realm::Machine::get_machine());
      pq.only_kind(Realm::Processor::LOC_PROC);
      for (Realm::Processor proc : pq) {
        handle.at(proc);
      }
    });
  }
}

} // namespace test
