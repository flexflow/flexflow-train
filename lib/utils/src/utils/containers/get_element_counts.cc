#include "utils/containers/get_element_counts.h"
#include "utils/archetypes/ordered_value_type.h"
#include "utils/containers/vector_of.h"

namespace FlexFlow {

using O_T = ordered_value_type<0>;

template std::map<O_T, positive_int>
    get_element_counts(std::vector<O_T> const &);
template std::map<O_T, positive_int>
    get_element_counts(std::multiset<O_T> const &);
template std::unordered_map<O_T, positive_int>
    get_element_counts(std::unordered_multiset<O_T> const &);

std::map<char, positive_int> get_element_counts(std::string const &s) {
  return get_element_counts(vector_of(s));
}

} // namespace FlexFlow
