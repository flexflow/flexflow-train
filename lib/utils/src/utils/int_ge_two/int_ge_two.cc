#include "utils/int_ge_two/int_ge_two.h"
#include <libassert/assert.hpp>

namespace FlexFlow {

int_ge_two::int_ge_two(int value) : value_(value) {
  this->check_invariant();
}

int_ge_two::int_ge_two(size_t value) : value_(value) {
  ASSERT(value == static_cast<size_t>(this->value_));

  this->check_invariant();
}

int_ge_two::int_ge_two(nonnegative_int value) : value_(value.unwrap_nonnegative()) {
  this->check_invariant();
}

int_ge_two::operator int() const noexcept {
  return this->value_;
}

int_ge_two::operator nonnegative_int() const noexcept {
  return nonnegative_int{this->value_};
}

int_ge_two::operator positive_int() const noexcept {
  return positive_int{this->value_};
}

bool int_ge_two::operator<(int_ge_two other) const {
  return this->value_ < other.value_;
}

bool int_ge_two::operator==(int_ge_two other) const {
  return this->value_ == other.value_;
}

bool int_ge_two::operator>(int_ge_two other) const {
  return this->value_ > other.value_;
}

bool int_ge_two::operator<=(int_ge_two other) const {
  return this->value_ <= other.value_;
}

bool int_ge_two::operator!=(int_ge_two other) const {
  return this->value_ != other.value_;
}

bool int_ge_two::operator>=(int_ge_two other) const {
  return this->value_ >= other.value_;
}

bool int_ge_two::operator<(positive_int other) const {
  return this->value_ < other.int_from_positive_int();
}

bool int_ge_two::operator==(positive_int other) const {
  return this->value_ == other.int_from_positive_int();
}

bool int_ge_two::operator>(positive_int other) const {
  return this->value_ > other.int_from_positive_int();
}

bool int_ge_two::operator<=(positive_int other) const {
  return this->value_ <= other.int_from_positive_int();
}

bool int_ge_two::operator!=(positive_int other) const {
  return this->value_ != other.int_from_positive_int();
}

bool int_ge_two::operator>=(positive_int other) const {
  return this->value_ >= other.int_from_positive_int();
}

bool operator<(positive_int lhs, int_ge_two rhs) {
  return lhs.int_from_positive_int() < rhs.value_;
}

bool operator==(positive_int lhs, int_ge_two rhs) {
  return lhs.int_from_positive_int() == rhs.value_;
}

bool operator>(positive_int lhs, int_ge_two rhs) {
  return lhs.int_from_positive_int() > rhs.value_;
}

bool operator<=(positive_int lhs, int_ge_two rhs) {
  return lhs.int_from_positive_int() <= rhs.value_;
}

bool operator!=(positive_int lhs, int_ge_two rhs) {
  return lhs.int_from_positive_int() != rhs.value_;
}

bool operator>=(positive_int lhs, int_ge_two rhs) {
  return lhs.int_from_positive_int() >= rhs.value_;
}

bool int_ge_two::operator<(nonnegative_int other) const {
  return this->value_ < other.unwrap_nonnegative();
}

bool int_ge_two::operator==(nonnegative_int other) const {
  return this->value_ == other.unwrap_nonnegative();
}

bool int_ge_two::operator>(nonnegative_int other) const {
  return this->value_ > other.unwrap_nonnegative();
}

bool int_ge_two::operator<=(nonnegative_int other) const {
  return this->value_ <= other.unwrap_nonnegative();
}

bool int_ge_two::operator!=(nonnegative_int other) const {
  return this->value_ != other.unwrap_nonnegative();
}

bool int_ge_two::operator>=(nonnegative_int other) const {
  return this->value_ >= other.unwrap_nonnegative();
}

bool operator<(nonnegative_int lhs, int_ge_two rhs) {
  return lhs.unwrap_nonnegative() < rhs.value_;
}

bool operator==(nonnegative_int lhs, int_ge_two rhs) {
  return lhs.unwrap_nonnegative() == rhs.value_;
}

bool operator>(nonnegative_int lhs, int_ge_two rhs) {
  return lhs.unwrap_nonnegative() > rhs.value_;
}

bool operator<=(nonnegative_int lhs, int_ge_two rhs) {
  return lhs.unwrap_nonnegative() <= rhs.value_;
}

bool operator!=(nonnegative_int lhs, int_ge_two rhs) {
  return lhs.unwrap_nonnegative() != rhs.value_;
}

bool operator>=(nonnegative_int lhs, int_ge_two rhs) {
  return lhs.unwrap_nonnegative() >= rhs.value_;
}

bool int_ge_two::operator<(int other) const {
  return this->value_ < other;
}

bool int_ge_two::operator==(int other) const {
  return this->value_ == other;
}

bool int_ge_two::operator>(int other) const {
  return this->value_ > other;
}

bool int_ge_two::operator<=(int other) const {
  return this->value_ <= other;
}

bool int_ge_two::operator!=(int other) const {
  return this->value_ != other;
}

bool int_ge_two::operator>=(int other) const {
  return this->value_ >= other;
}

bool operator<(int lhs, int_ge_two rhs) {
  return lhs < rhs.value_;
}

bool operator==(int lhs, int_ge_two rhs) {
  return lhs == rhs.value_;
}

bool operator>(int lhs, int_ge_two rhs) {
  return lhs > rhs.value_;
}

bool operator<=(int lhs, int_ge_two rhs) {
  return lhs <= rhs.value_;
}

bool operator!=(int lhs, int_ge_two rhs) {
  return lhs != rhs.value_;
}

bool operator>=(int lhs, int_ge_two rhs) {
  return lhs >= rhs.value_;
}

int_ge_two int_ge_two::operator+(int_ge_two other) const {
  return int_ge_two{
    this->value_ + other.value_,
  };
}

int_ge_two int_ge_two::operator+(positive_int other) const {
  return int_ge_two{
    this->value_ + other.int_from_positive_int(),
  };
}

int_ge_two int_ge_two::operator+(nonnegative_int other) const {
  return int_ge_two{
    this->value_ + other.unwrap_nonnegative(),
  };
}

int_ge_two &int_ge_two::operator++() {
  this->value_++;
  this->check_invariant();
  return *this;
}

int_ge_two int_ge_two::operator++(int) {
  int_ge_two result = *this;
  this->value_++;
  this->check_invariant();
  return result;
}

int_ge_two &int_ge_two::operator+=(int_ge_two other) {
  this->value_ += other.value_;
  this->check_invariant();
  return *this;
}

int_ge_two &int_ge_two::operator+=(positive_int other) {
  this->value_ += other.int_from_positive_int();
  this->check_invariant();
  return *this;
}

int_ge_two &int_ge_two::operator+=(nonnegative_int other) {
  this->value_ += other.unwrap_nonnegative();
  this->check_invariant();
  return *this;
}

int_ge_two operator+(nonnegative_int lhs, int_ge_two rhs) {
  return rhs + lhs;
}

int_ge_two operator+(positive_int lhs, int_ge_two rhs) {
  return rhs + lhs;
}

int_ge_two int_ge_two::operator*(int_ge_two other) const {
  return int_ge_two{
    this->value_ * other.value_,
  };
}

int_ge_two &int_ge_two::operator*=(int_ge_two other) {
  this->value_ *= other.value_;
  this->check_invariant();
  return *this;
}

int_ge_two int_ge_two::operator*(positive_int other) const {
  return int_ge_two{
    this->value_ * other.int_from_positive_int(),
  };
}

int_ge_two &int_ge_two::operator*=(positive_int other) {
  this->value_ *= other.int_from_positive_int();
  this->check_invariant();
  return *this;
}

nonnegative_int int_ge_two::operator*(nonnegative_int other) const {
  return nonnegative_int{
    this->value_ * other.unwrap_nonnegative(),
  };
}

int_ge_two operator*(positive_int lhs, int_ge_two rhs) {
  return rhs * lhs;
}

nonnegative_int operator*(nonnegative_int lhs, int_ge_two rhs) {
  return rhs * lhs;
}

int int_ge_two::int_from_int_ge_two() const {
  return this->value_;
}

nonnegative_int int_ge_two::nonnegative_int_from_int_ge_two() const {
  return nonnegative_int{
    this->value_,
  };
}

positive_int int_ge_two::positive_int_from_int_ge_two() const {
  return positive_int{
    this->value_,
  };
}

std::ostream &operator<<(std::ostream &os, int_ge_two n) {
  return (os << fmt::to_string(n));
}

int format_as(int_ge_two x) {
  return x.value_;
}

void int_ge_two::check_invariant() const {
  ASSERT(this->value_ >= 2);
}

int_ge_two operator""_ge2(unsigned long long int x) {
  ASSERT(x <=
         static_cast<unsigned long long int>(std::numeric_limits<int>::max()));

  return int_ge_two{static_cast<int>(x)};
}

} // namespace FlexFlow

namespace nlohmann {
::FlexFlow::int_ge_two
    adl_serializer<::FlexFlow::int_ge_two>::from_json(json const &j) {
  return ::FlexFlow::int_ge_two{j.template get<int>()};
}

void adl_serializer<::FlexFlow::int_ge_two>::to_json(
    json &j, ::FlexFlow::int_ge_two t) {
  j = t.int_from_int_ge_two();
}
} // namespace nlohmann

namespace rc {
Gen<::FlexFlow::int_ge_two> Arbitrary<::FlexFlow::int_ge_two>::arbitrary() {
  return gen::construct<::FlexFlow::int_ge_two>(gen::positive<int>());
}
} // namespace rc

namespace std {
std::size_t hash<::FlexFlow::int_ge_two>::operator()(
    FlexFlow::int_ge_two n) const noexcept {
  return std::hash<int>{}(n.int_from_int_ge_two());
}

} // namespace std
