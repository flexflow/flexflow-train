#include "utils/orthotope/dim_domain.h"
#include "test/utils/doctest/fmt/unordered_set.h"
#include "utils/orthotope/dim_ordering.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("get_domain_dims") {
    DimDomain<int> domain = DimDomain<int>{{
        {3, 5_p},
        {7, 2_p},
        {1, 3_p},
    }};

    std::unordered_set<int> result = get_domain_dims(domain);
    std::unordered_set<int> correct = {
        3,
        7,
        1,
    };

    CHECK(result == correct);
  }

  TEST_CASE("restrict_domain_to_dims") {
    DimDomain<int> domain = DimDomain<int>{{
        {3, 5_p},
        {7, 2_p},
        {1, 3_p},
    }};

    SUBCASE("allowed is a strict subset of the dims") {
      std::unordered_set<int> allowed = {
          3,
          1,
      };

      DimDomain<int> result = restrict_domain_to_dims(domain, allowed);
      DimDomain<int> correct = DimDomain<int>{{
          {3, 5_p},
          {1, 3_p},
      }};

      CHECK(result == correct);
    }

    SUBCASE("allowed is the same as dims") {
      std::unordered_set<int> allowed = {
          3,
          7,
          1,
      };

      DimDomain<int> result = restrict_domain_to_dims(domain, allowed);
      DimDomain<int> correct = domain;

      CHECK(result == correct);
    }

    SUBCASE("allowed is empty") {
      std::unordered_set<int> allowed = {};

      DimDomain<int> result = restrict_domain_to_dims(domain, allowed);
      DimDomain<int> correct = DimDomain<int>{{}};

      CHECK(result == correct);
    }

    SUBCASE("allowed is mutually exclusive with dims") {
      std::unordered_set<int> allowed = {
          6,
          8,
      };

      DimDomain<int> result = restrict_domain_to_dims(domain, allowed);
      DimDomain<int> correct = DimDomain<int>{{}};

      CHECK(result == correct);
    }

    SUBCASE("allowed is overlapping with dims") {
      std::unordered_set<int> allowed = {
          6,
          8,
          7,
      };

      DimDomain<int> result = restrict_domain_to_dims(domain, allowed);
      DimDomain<int> correct = DimDomain<int>{{
          {7, 2_p},
      }};

      CHECK(result == correct);
    }

    SUBCASE("allowed is a superset of dims") {
      std::unordered_set<int> allowed = {
          6,
          8,
          7,
          3,
          1,
      };

      DimDomain<int> result = restrict_domain_to_dims(domain, allowed);
      DimDomain<int> correct = domain;

      CHECK(result == correct);
    }
  }

  TEST_CASE("orthotope_from_dim_domain") {
    DimDomain<int> domain = DimDomain<int>{{
        {3, 5_p},
        {7, 2_p},
        {1, 3_p},
    }};

    Orthotope result =
        orthotope_from_dim_domain(domain, make_default_dim_ordering<int>());
    Orthotope correct = Orthotope{{
        3_p,
        5_p,
        2_p,
    }};

    CHECK(result == correct);
  }
  
  TEST_CASE("dim_domain_from_orthotope") {
    Orthotope orthotope = Orthotope{{
        3_p,
        5_p,
        2_p,
    }};

    std::unordered_set<int> dims = {
      3, 7, 1
    };

    DimDomain<int> result = dim_domain_from_orthotope(orthotope, dims, make_default_dim_ordering<int>());

    DimDomain<int> correct = DimDomain<int>{{
        {3, 5_p},
        {7, 2_p},
        {1, 3_p},
    }};
        
    CHECK(result == correct);
  }
}
