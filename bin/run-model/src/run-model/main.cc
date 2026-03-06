#include "realm-execution/realm_context.h"
#include "realm-execution/realm_manager.h"
#include "utils/cli/cli_get_help_message.h"
#include "utils/cli/cli_parse.h"
#include "utils/cli/cli_parse_result.h"
#include "utils/cli/cli_spec.h"
#include "utils/nonnegative_int/nonnegative_int.h"
#include "utils/positive_int/positive_int.h"
#include <string_view>

using namespace FlexFlow;

static char *leak_string_contents(std::string_view str) {
  // Realm command-line arguments require char* so intentionally leak the
  // allocated string contents here
  std::vector<char> *content = new std::vector<char>{str.begin(), str.end()};
  content->push_back(0); // NUL byte
  return content->data();
}

static std::vector<char *> make_realm_args(std::string_view executable_name) {
  std::vector<char *> result;
  result.push_back(leak_string_contents(executable_name));
  return result;
}

int main(int argc, char **argv) {
  CLISpec cli = empty_cli_spec();

  CLIArgumentKey arg_key_help = cli_add_help_flag(cli);

  CLIArgumentKey key_mapped_pcg_json_file = cli_add_positional_argument(
      cli,
      CLIPositionalArgumentSpec{
          "mapped_pcg_json",
          std::nullopt,
          "path to a file containing mappped PCG encoded as JSON"});

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
      return 1;
    }

    result.value();
  });

  bool help = cli_get_flag(parsed, arg_key_help);
  if (help) {
    std::cerr << cli_get_help_message(prog_name, cli);
    return 1;
  }

  std::vector<char *> realm_args = make_realm_args(prog_name);
  int realm_argc = realm_args.size();
  char **realm_argv = realm_args.data();
  RealmManager manager(&realm_argc, &realm_argv);

  FlexFlow::Realm::Event event =
      manager.start_controller([](RealmContext &ctx) {});
  event.wait();

  return 0;
}
