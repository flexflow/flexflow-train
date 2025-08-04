#include "utils/units/milliseconds_t.h"
#include "utils/hash-utils.h"
#include <fmt/format.h>
#include <libassert/assert.hpp>
#include <limits>

namespace FlexFlow {

milliseconds_t::milliseconds_t(float value) : value(value) {}

bool milliseconds_t::operator<(milliseconds_t const &other) const {
  return this->value < other.value;
}

bool milliseconds_t::operator==(milliseconds_t const &other) const {
  return this->value == other.value;
}

bool milliseconds_t::operator>(milliseconds_t const &other) const {
  return this->value > other.value;
}

bool milliseconds_t::operator<=(milliseconds_t const &other) const {
  return this->value <= other.value;
}

bool milliseconds_t::operator!=(milliseconds_t const &other) const {
  return this->value != other.value;
}

bool milliseconds_t::operator>=(milliseconds_t const &other) const {
  return this->value >= other.value;
}

milliseconds_t milliseconds_t::operator+(milliseconds_t const &other) const {
  return milliseconds_t{
      this->value + other.value,
  };
}

float milliseconds_t::unwrap_milliseconds() const {
  return this->value;
}

milliseconds_t operator""_ms(long double x) {
  ASSERT(x <= std::numeric_limits<float>::max());
  ASSERT(x >= std::numeric_limits<float>::lowest());
  return milliseconds_t{static_cast<float>(x)};
}

milliseconds_t operator""_ms(unsigned long long int x) {
  ASSERT(x <= std::numeric_limits<float>::max());
  return milliseconds_t{static_cast<float>(x)};
}

std::ostream &operator<<(std::ostream &s, milliseconds_t const &m) {
  return (s << fmt::to_string(m));
}

std::string format_as(milliseconds_t const &m) {
  return fmt::format("{}_ms", m.unwrap_milliseconds());
}

} // namespace FlexFlow

namespace nlohmann {
::FlexFlow::milliseconds_t
    adl_serializer<::FlexFlow::milliseconds_t>::from_json(json const &j) {
  return ::FlexFlow::milliseconds_t{j.template get<float>()};
}

void adl_serializer<::FlexFlow::milliseconds_t>::to_json(
    json &j, ::FlexFlow::milliseconds_t t) {
  j = t.unwrap_milliseconds();
}
} // namespace nlohmann

namespace rc {

Gen<::FlexFlow::milliseconds_t>
    Arbitrary<::FlexFlow::milliseconds_t>::arbitrary() {
  return gen::construct<::FlexFlow::milliseconds_t>(gen::arbitrary<float>());
}

} // namespace rc

namespace std {

size_t hash<::FlexFlow::milliseconds_t>::operator()(
    ::FlexFlow::milliseconds_t const &m) const noexcept {
  return ::FlexFlow::get_std_hash(m.unwrap_milliseconds());
}

} // namespace std
