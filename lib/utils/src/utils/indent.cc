#include "utils/indent.h"
#include "utils/containers/flatmap.h"

namespace FlexFlow {

std::string indent(std::string const &s, int indent_size) {
  std::string indent_str(indent_size, ' ');
  return indent_str + flatmap(s, [&](char c) -> std::string {
           if (c == '\n') {
             return "\n" + indent_str;
           } else {
             return std::string{c};
           };
         });
}

} // namespace FlexFlow
