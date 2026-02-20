#include "utils/disjoint_set.h"
#include "utils/archetypes/value_type.h"

namespace FlexFlow {

using T = value_type<0>;

template class m_disjoint_set<T>;

} // namespace FlexFlow
