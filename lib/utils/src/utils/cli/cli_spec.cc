#include "utils/cli/cli_spec.h"
#include "utils/containers/count.h"
#include "utils/containers/transform.h"
#include "utils/integer_conversions.h"
#include "utils/nonnegative_int/nonnegative_range.h"
#include "utils/nonnegative_int/num_elements.h"

namespace FlexFlow {

CLISpec empty_cli_spec() {
  return CLISpec{{}, {}};
}

std::vector<CLIFlagKey> cli_get_flag_keys(CLISpec const &cli) {
  return transform(nonnegative_range(num_elements(cli.flags)),
                   [](nonnegative_int idx) { return CLIFlagKey{idx}; });
}

CLIArgumentKey cli_add_help_flag(CLISpec &cli) {
  CLIFlagSpec help_flag =
      CLIFlagSpec{"help", 'h', "show this help message and exit"};
  return cli_add_flag(cli, help_flag);
}

CLIArgumentKey cli_add_flag(CLISpec &cli, CLIFlagSpec const &flag_spec) {
  CLIArgumentKey key = CLIArgumentKey{CLIFlagKey{num_elements(cli.flags)}};
  cli.flags.push_back(flag_spec);
  return key;
}

CLIArgumentKey
    cli_add_positional_argument(CLISpec &cli,
                                CLIPositionalArgumentSpec const &arg) {
  CLIArgumentKey key = CLIArgumentKey{
      CLIPositionalArgumentKey{num_elements(cli.positional_arguments)}};
  cli.positional_arguments.push_back(arg);
  return key;
}

} // namespace FlexFlow
