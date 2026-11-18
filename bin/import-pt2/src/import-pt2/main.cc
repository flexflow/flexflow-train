#include "import-pt2/pt2_archive.h"
#include "import-pt2/pt2_to_computation_graph.h"
#include "pcg/computation_graph.h"
#include "pcg/file_format/v1/v1_computation_graph.h"
#include "utils/cli/cli_get_help_message.h"
#include "utils/cli/cli_parse.h"
#include "utils/cli/cli_parse_result.h"
#include "utils/cli/cli_spec.h"
#include <iostream>

using namespace ::FlexFlow;

int main(int argc, char **argv) {
  CLISpec cli;

  CLIArgumentKey key_help = cli_add_help_flag(cli);

  CLIArgumentKey key_dot = cli.add_flag(CLIFlagSpec{
      /*long_flag=*/"dot",
      /*short_flag=*/std::nullopt,
      /*description=*/
      "output a dot representation of the imported computation graph "
      "instead of JSON",
  });

  CLIArgumentKey key_pt2_path =
      cli.add_positional_argument(CLIPositionalArgumentSpec{
          /*name=*/"pt2_file",
          /*choices=*/std::nullopt,
          /*description=*/"a .pt2 file written by torch.export.save()",
      });

  assert(argc >= 1);
  std::string prog_name = argv[0];

  CLIParseResult parsed = ({
    tl::expected<CLIParseResult, std::string> result =
        cli_parse(cli, argc, argv);
    if (!result.has_value()) {
      std::cerr << cli_get_help_message(prog_name, cli);
      std::cerr << std::endl;
      std::cerr << "error: " << result.error() << std::endl;
      return 1;
    }

    result.value();
  });

  if (cli_get_flag(parsed, key_help)) {
    std::cerr << cli_get_help_message(prog_name, cli);
    return 1;
  }

  std::string pt2_path = cli_get_positional_argument(parsed, key_pt2_path);
  bool dot = cli_get_flag(parsed, key_dot);

  ComputationGraph cg = ({
    std::optional<ComputationGraph> result;
    try {
      result = computation_graph_from_pt2_model_json(
          read_pt2_model_json(pt2_path));
    } catch (std::exception const &e) {
      std::cerr << "error: " << e.what() << std::endl;
      return 1;
    }
    result.value();
  });

  if (dot) {
    std::cout << as_dot(cg) << std::endl;
  } else {
    nlohmann::json json_output = to_v1(cg);
    std::cout << json_output.dump(2) << std::endl;
  }

  return 0;
}
