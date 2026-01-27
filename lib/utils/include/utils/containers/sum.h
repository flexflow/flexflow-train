#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_SUM_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_SUM_H

#include <optional>

namespace FlexFlow {

/**
 * @details An empty container vacuously has sum 0
 **/
template <typename Container, typename Element = typename Container::value_type>
Element sum(Container const &container) {
  std::optional<Element> result;
  for (Element const &element : container) {
    if (result.has_value()) {
      result.value() += element;
    } else {
      result = element;
    }
  }

  if (result.has_value()) {
    return result.value();
  } else {
    return Element{0};
  }
}

} // namespace FlexFlow

#endif
