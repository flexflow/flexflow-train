#include "utils/random_utils.h"
#include "utils/archetypes/value_type.h"

namespace FlexFlow {

float randf() {
  return static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX);
}

using T = value_type<0>;

template T select_random(std::vector<T> const &);
template T select_random_determistic(std::vector<T> const &,
                                     std::vector<float> const &,
                                     float);
template T select_random(std::vector<T> const &, std::vector<float> const &);

} // namespace FlexFlow
