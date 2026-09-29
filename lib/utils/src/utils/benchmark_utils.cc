#include "utils/benchmark_utils.h"
#include "utils/cli/cli_get_help_message.h"
#include "utils/cli/cli_parse.h"
#include "utils/cli/cli_parse_result.h"
#include "utils/cli/cli_spec.h"
#include "utils/containers/keys.h"
#include "utils/containers/sorted.h"
#include "utils/optional.h"
#include <libassert/assert.hpp>

namespace {

void check_header_builds(bool dry_run) {
  int x = 0;
  LOOP(10, dry_run) {
    x++;
    x++;
  }
}

} // namespace

namespace FlexFlow {

void benchmark_main(
    int argc,
    char **argv,
    std::map<std::string, std::function<void(bool)>> const &benchmarks) {
  CLISpec cli;

  CLIArgumentKey arg_key_help = cli_add_help_flag(cli);
  CLIArgumentKey arg_key_list = cli.add_flag(CLIFlagSpec{
      /*long_flag=*/"list",
      /*short_flag=*/'l',
      /*description=*/"list benchmarks and exit",
  });
  CLIArgumentKey arg_key_dry_run = cli.add_flag(CLIFlagSpec{
      /*long_flag=*/"dry-run",
      /*short_flag=*/std::nullopt,
      /*description=*/
      "do all the parsing and setup but exit just before running the benchmark",
  });

  CLIArgumentKey key_benchmark = cli.add_named_argument(CLINamedArgumentSpec{
      /*long_flag=*/"benchmark",
      /*metavar=*/"BENCHMARK",
      /*choices=*/sorted(keys(benchmarks)),
      /*description=*/"benchmark name",
  });

  ASSERT(argc >= 1);
  std::string prog_name = argv[0];

  CLIParseResult parsed = ({
    tl::expected<CLIParseResult, std::string> result =
        cli_parse(cli, argc, argv);
    if (!result.has_value()) {
      std::string error_msg = result.error();
      std::cerr << cli_get_help_message(prog_name, cli);
      std::cerr << std::endl;
      std::cerr << "error: " << error_msg << std::endl;
      std::exit(1);
    }

    result.value();
  });

  bool help = cli_get_flag(parsed, arg_key_help);
  if (help) {
    std::cerr << cli_get_help_message(prog_name, cli);
    std::exit(1);
  }

  bool list_benchmarks = cli_get_flag(parsed, arg_key_list);
  if (list_benchmarks) {
    for (std::string benchmark_name : keys(benchmarks)) {
      std::cout << benchmark_name << std::endl;
    }
    std::exit(0);
  }

  std::string benchmark_name =
      assert_unwrap(cli_get_named_argument(parsed, key_benchmark));

  std::function<void(bool)> benchmark_func = benchmarks.at(benchmark_name);

  bool dry_run = cli_get_flag(parsed, arg_key_dry_run);
  benchmark_func(dry_run);
}

} // namespace FlexFlow
