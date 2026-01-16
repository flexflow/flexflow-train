#include "test/utils/doctest/check_kv.h"
#include "utils/indent.h"
#include <sstream>

namespace FlexFlow {

std::string check_kv(std::string const &k, std::string const &v) {
  std::ostringstream oss;

  oss << std::endl
      << indent(k + "=", /*indent_size=*/4) << std::endl
      << indent(v, /*indent_size=*/6);

  return oss.str();
}

} // namespace FlexFlow
