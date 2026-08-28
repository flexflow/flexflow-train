#include "realm-execution/redops/redop_id_t.h"
#include "test/utils/rapidcheck/doctest.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("get_realm_reduction_op_id_for_redop_id") {
    RC_SUBCASE("does not produce Realm::ReductionOpID value 0",
               [](redop_id_t redop) {
                 FlexFlow::Realm::ReductionOpID result =
                     get_realm_reduction_op_id_for_redop_id(
                         redop_id_t::SUM_BOOL_REDOP_ID);

                 // Realm::ReductionOpID = 0 has a special meaning of "absence of a reduction op"
                 return (result > 0);
               });
  }
}
