#include "utils/containers/concat_vectors.h"
#include "utils/archetypes/value_type.h"

namespace FlexFlow {

using T = value_type<0>;

template std::vector<T> concat_vectors(std::vector<T> const &,
                                       std::vector<T> const &);

template std::vector<T> concat_vectors(std::vector<std::vector<T>> const &);

} // namespace FlexFlow
