#include "utils/indent.h"
#include "utils/containers/flatmap.h"

namespace FlexFlow {

std::string indent(std::string const &s) {
  return "  " + flatmap(s, [](char c) -> std::string {
           if (c == '\n') {
             return "\n  ";
           } else {
             return std::string{c};
           };
         });
}

} // namespace FlexFlow
