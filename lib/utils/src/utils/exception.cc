#include "utils/exception.h"

namespace FlexFlow {

not_implemented::not_implemented(std::string const &function_name,
                                 std::string const &file_name,
                                 int line)
    : std::logic_error(fmt::format("Function '{}' not yet implemented at {}:{}",
                                   function_name,
                                   file_name,
                                   line)){};

std::runtime_error mk_runtime_error(std::string const &msg) {
  return std::runtime_error(msg);
}

} // namespace FlexFlow
