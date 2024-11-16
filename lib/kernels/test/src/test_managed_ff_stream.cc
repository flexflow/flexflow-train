#include "doctest/doctest.h"
#include "kernels/managed_ff_stream.h"

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("ManagedFFStream") {
    ManagedFFStream base_stream{};
    ffStream_t const *base_stream_ptr = &base_stream.raw_stream();

    SUBCASE("move constructor") {
      ManagedFFStream new_stream(std::move(base_stream));
      CHECK(&base_stream.raw_stream() == nullptr);
      CHECK(&new_stream.raw_stream() == base_stream_ptr);
    }

    SUBCASE("move assignment operator") {
      SUBCASE("move assign to other") {
        ManagedFFStream new_stream{};
        new_stream = std::move(base_stream);
        CHECK(&base_stream.raw_stream() == nullptr);
        CHECK(&new_stream.raw_stream() == base_stream_ptr);
      }

      SUBCASE("move assign to self") {
        base_stream = std::move(base_stream);
        CHECK(&base_stream.raw_stream() == base_stream_ptr);
      }
    }
  }
}
