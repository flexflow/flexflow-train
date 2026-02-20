#include "utils/units/bytes_per_second_t.h"
#include "utils/hash-utils.h"
#include <cmath>
#include <fmt/format.h>
#include <libassert/assert.hpp>
#include <limits>

namespace FlexFlow {

bytes_per_second_t::bytes_per_second_t(float value) : value(value) {
  ASSERT(std::isfinite(value));
}

bool bytes_per_second_t::operator<(bytes_per_second_t const &other) const {
  return this->value < other.value;
}

bool bytes_per_second_t::operator==(bytes_per_second_t const &other) const {
  return this->value == other.value;
}

bool bytes_per_second_t::operator>(bytes_per_second_t const &other) const {
  return this->value > other.value;
}

bool bytes_per_second_t::operator<=(bytes_per_second_t const &other) const {
  return this->value <= other.value;
}

bool bytes_per_second_t::operator!=(bytes_per_second_t const &other) const {
  return this->value != other.value;
}

bool bytes_per_second_t::operator>=(bytes_per_second_t const &other) const {
  return this->value >= other.value;
}

bytes_per_second_t
    bytes_per_second_t::operator+(bytes_per_second_t const &other) const {
  return bytes_per_second_t{
      this->value + other.value,
  };
}

milliseconds_t operator/(num_bytes_t num_bytes,
                         bytes_per_second_t bytes_per_second) {
  int raw_num_bytes = num_bytes.unwrap_num_bytes().unwrap_nonnegative();
  float raw_bytes_per_millisecond =
      bytes_per_second.unwrap_bytes_per_second() * 1000;

  return milliseconds_t{raw_num_bytes / raw_bytes_per_millisecond};
}

float bytes_per_second_t::unwrap_bytes_per_second() const {
  return this->value;
}

std::ostream &operator<<(std::ostream &s, bytes_per_second_t const &m) {
  return (s << fmt::to_string(m));
}

std::string format_as(bytes_per_second_t const &m) {
  return fmt::format("{}_bytes/s", m.unwrap_bytes_per_second());
}

} // namespace FlexFlow

namespace nlohmann {
::FlexFlow::bytes_per_second_t
    adl_serializer<::FlexFlow::bytes_per_second_t>::from_json(json const &j) {
  return ::FlexFlow::bytes_per_second_t{j.template get<float>()};
}

void adl_serializer<::FlexFlow::bytes_per_second_t>::to_json(
    json &j, ::FlexFlow::bytes_per_second_t t) {
  j = t.unwrap_bytes_per_second();
}
} // namespace nlohmann

namespace rc {

Gen<::FlexFlow::bytes_per_second_t>
    Arbitrary<::FlexFlow::bytes_per_second_t>::arbitrary() {
  return gen::construct<::FlexFlow::bytes_per_second_t>(
      gen::arbitrary<float>());
}

} // namespace rc

namespace std {

size_t hash<::FlexFlow::bytes_per_second_t>::operator()(
    ::FlexFlow::bytes_per_second_t const &m) const noexcept {
  return ::FlexFlow::get_std_hash(m.unwrap_bytes_per_second());
}

} // namespace std
