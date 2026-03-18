#include "internal/realm_test_utils.h"
#include <fmt/format.h>
#include <string>

namespace FlexFlow {

static char *leak_string_contents(std::string const &str) {
  // Realm command-line arguments require char* so intentionally leak the
  // allocated string contents here
  std::vector<char> *content = new std::vector<char>{str.begin(), str.end()};
  content->push_back(0); // NUL byte
  return content->data();
}

std::vector<char *> make_fake_realm_args(positive_int num_cpus,
                                         nonnegative_int num_gpus) {
  std::vector<char *> result;
  result.push_back(leak_string_contents("fake_executable_name"));
  result.push_back(leak_string_contents("-ll:cpu"));
  result.push_back(leak_string_contents(fmt::to_string(num_cpus)));
  if (num_gpus > 0) {
    result.push_back(leak_string_contents("-ll:gpu"));
    result.push_back(leak_string_contents(fmt::to_string(num_gpus)));
  }
  return result;
}

} // namespace FlexFlow
