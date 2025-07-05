#include "utils/units/num_bytes_t.h"
#include "utils/hash-utils.h"
#include <fmt/format.h>
#include <libassert/assert.hpp>
#include <limits>

namespace FlexFlow {

num_bytes_t::num_bytes_t(nonnegative_int value) : value(value) {}

bool num_bytes_t::operator<(num_bytes_t const &other) const {
  return this->value < other.value;
}

bool num_bytes_t::operator==(num_bytes_t const &other) const {
  return this->value == other.value;
}

bool num_bytes_t::operator>(num_bytes_t const &other) const {
  return this->value > other.value;
}

bool num_bytes_t::operator<=(num_bytes_t const &other) const {
  return this->value <= other.value;
}

bool num_bytes_t::operator!=(num_bytes_t const &other) const {
  return this->value != other.value;
}

bool num_bytes_t::operator>=(num_bytes_t const &other) const {
  return this->value >= other.value;
}

num_bytes_t num_bytes_t::operator+(num_bytes_t const &other) const {
  return num_bytes_t{
      this->value + other.value,
  };
}

nonnegative_int num_bytes_t::unwrap_num_bytes() const {
  return this->value;
}

num_bytes_t operator""_bytes(unsigned long long int x) {
  return num_bytes_t{nonnegative_int{x}};
}

std::ostream &operator<<(std::ostream &s, num_bytes_t const &m) {
  return (s << fmt::to_string(m));
}

std::string format_as(num_bytes_t const &m) {
  return fmt::format("{}_bytes", m.unwrap_num_bytes());
}

} // namespace FlexFlow

namespace nlohmann {
::FlexFlow::num_bytes_t
    adl_serializer<::FlexFlow::num_bytes_t>::from_json(json const &j) {
  return ::FlexFlow::num_bytes_t{j.template get<::FlexFlow::nonnegative_int>()};
}

void adl_serializer<::FlexFlow::num_bytes_t>::to_json(
    json &j, ::FlexFlow::num_bytes_t t) {
  j = t.unwrap_num_bytes();
}
} // namespace nlohmann

namespace rc {

Gen<::FlexFlow::num_bytes_t> Arbitrary<::FlexFlow::num_bytes_t>::arbitrary() {
  return gen::construct<::FlexFlow::num_bytes_t>(
      gen::arbitrary<::FlexFlow::nonnegative_int>());
}

} // namespace rc

namespace std {

size_t hash<::FlexFlow::num_bytes_t>::operator()(
    ::FlexFlow::num_bytes_t const &m) const noexcept {
  return ::FlexFlow::get_std_hash(m.unwrap_num_bytes());
}

} // namespace std
