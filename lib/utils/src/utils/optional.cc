#include "utils/optional.h"
#include "utils/archetypes/value_type.h"

namespace FlexFlow {

using T = value_type<0>;

template T or_else(std::optional<T> const &, std::function<T()> &&);
template T const &unwrap(std::optional<T> const &,
                         std::function<void()> const &);
template T const &assert_unwrap(std::optional<T> const &);

} // namespace FlexFlow
