#ifndef _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_INITIALIZERS_UNIFORM_INITIALIZER_ATTRS_H
#define _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_INITIALIZERS_UNIFORM_INITIALIZER_ATTRS_H

#include "op-attrs/initializers/uniform_initializer_attrs.dtg.h"
#include <rapidcheck.h>

namespace rc {

template <>
struct Arbitrary<::FlexFlow::UniformInitializerAttrs> {
  static Gen<::FlexFlow::UniformInitializerAttrs> arbitrary();
};

} // namespace rc

#endif
