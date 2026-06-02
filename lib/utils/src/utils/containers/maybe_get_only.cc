#include "utils/containers/maybe_get_only.h"
#include "utils/archetypes/value_type.h"
#include <vector>

namespace FlexFlow {

template std::optional<value_type<0>>
    maybe_get_only(std::vector<value_type<0>> const &);

} // namespace FlexFlow
