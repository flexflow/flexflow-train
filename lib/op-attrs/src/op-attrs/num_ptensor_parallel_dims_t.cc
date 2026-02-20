#include "op-attrs/num_ptensor_parallel_dims_t.h"
#include "utils/hash-utils.h"
#include <fmt/format.h>
#include <libassert/assert.hpp>
#include <limits>

namespace FlexFlow {

num_ptensor_parallel_dims_t::num_ptensor_parallel_dims_t(int value)
    : value(value) {
  this->check_invariant();
}

num_ptensor_parallel_dims_t::num_ptensor_parallel_dims_t(nonnegative_int value)
    : value(value.unwrap_nonnegative()) {}

num_ptensor_parallel_dims_t::num_ptensor_parallel_dims_t(positive_int value)
    : value(value.int_from_positive_int()) {}

bool num_ptensor_parallel_dims_t::operator<(
    num_ptensor_parallel_dims_t const &other) const {
  return this->value < other.value;
}

bool num_ptensor_parallel_dims_t::operator==(
    num_ptensor_parallel_dims_t const &other) const {
  return this->value == other.value;
}

bool num_ptensor_parallel_dims_t::operator>(
    num_ptensor_parallel_dims_t const &other) const {
  return this->value > other.value;
}

bool num_ptensor_parallel_dims_t::operator<=(
    num_ptensor_parallel_dims_t const &other) const {
  return this->value <= other.value;
}

bool num_ptensor_parallel_dims_t::operator!=(
    num_ptensor_parallel_dims_t const &other) const {
  return this->value != other.value;
}

bool num_ptensor_parallel_dims_t::operator>=(
    num_ptensor_parallel_dims_t const &other) const {
  return this->value >= other.value;
}

int num_ptensor_parallel_dims_t::int_from_num_ptensor_parallel_dims() const {
  return this->value;
}

nonnegative_int num_ptensor_parallel_dims_t::
    nonnegative_int_from_num_ptensor_parallel_dims() const {
  return nonnegative_int{this->value};
}

positive_int
    num_ptensor_parallel_dims_t::positive_int_from_num_ptensor_parallel_dims()
        const {
  return positive_int{this->value};
}

void num_ptensor_parallel_dims_t::check_invariant() const {
  ASSERT(this->value >= 2);
  ASSERT(this->value <= MAX_TENSOR_DIM + 2);
}

std::ostream &operator<<(std::ostream &s,
                         num_ptensor_parallel_dims_t const &m) {
  return (s << fmt::to_string(m));
}

std::string format_as(num_ptensor_parallel_dims_t const &m) {
  return fmt::format("{} parallel dims",
                     m.int_from_num_ptensor_parallel_dims());
}

} // namespace FlexFlow

namespace nlohmann {
::FlexFlow::num_ptensor_parallel_dims_t
    adl_serializer<::FlexFlow::num_ptensor_parallel_dims_t>::from_json(
        json const &j) {
  return ::FlexFlow::num_ptensor_parallel_dims_t{j.template get<int>()};
}

void adl_serializer<::FlexFlow::num_ptensor_parallel_dims_t>::to_json(
    json &j, ::FlexFlow::num_ptensor_parallel_dims_t t) {
  j = t.int_from_num_ptensor_parallel_dims();
}
} // namespace nlohmann

namespace rc {

Gen<::FlexFlow::num_ptensor_parallel_dims_t>
    Arbitrary<::FlexFlow::num_ptensor_parallel_dims_t>::arbitrary() {
  return gen::construct<::FlexFlow::num_ptensor_parallel_dims_t>(
      gen::arbitrary<float>());
}

} // namespace rc

namespace std {

size_t hash<::FlexFlow::num_ptensor_parallel_dims_t>::operator()(
    ::FlexFlow::num_ptensor_parallel_dims_t const &m) const noexcept {
  return ::FlexFlow::get_std_hash(m.int_from_num_ptensor_parallel_dims());
}

} // namespace std
