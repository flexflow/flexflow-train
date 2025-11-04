#include "utils/join_strings.h"
#include <vector>
#include "utils/archetypes/value_type.h"

namespace FlexFlow {

using T = value_type<0>;

template std::string join_strings(
  typename std::vector<T>::const_iterator,
  typename std::vector<T>::const_iterator,
  std::string const &,
  std::function<std::string(T const &)> const &);

template std::string join_strings(
  typename std::vector<std::string>::const_iterator, 
  typename std::vector<std::string>::const_iterator, 
  std::string const &);

template std::string join_strings(std::vector<std::string> const &, std::string const &);

template std::string join_strings(
  std::vector<T> const &, 
  std::string const &, 
  std::function<std::string(T const &)> const &);

} // namespace FlexFlow
