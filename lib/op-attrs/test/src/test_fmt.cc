#include "op-attrs/operator_attrs.h"
#include "test/utils/all.h"

using namespace FlexFlow;

TEST_CASE("ComputationGraphAttrs is fmtable") {
  rc::dc_check(
      [](ComputationGraphAttrs const &a) { CHECK(fmt::to_string(a) != ""); });
}