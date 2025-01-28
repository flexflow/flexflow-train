#include "utils/nonnegative_int/nonnegative_int.h"

namespace FlexFlow {

nonnegative_int::nonnegative_int(int value) {
  if (value < 0) {
    throw std::invalid_argument(
        "Value of nonnegative_int type must be nonnegative.");
  }
  this->value_ = value;
}

nonnegative_int::operator int() const noexcept {
  return this->value_;
}

bool nonnegative_int::operator<(nonnegative_int const &other) const {
  return this->value_ < other.value_;
}
bool nonnegative_int::operator==(nonnegative_int const &other) const {
  return this->value_ == other.value_;
}
bool nonnegative_int::operator>(nonnegative_int const &other) const {
  return this->value_ > other.value_;
}
bool nonnegative_int::operator<=(nonnegative_int const &other) const {
  return this->value_ <= other.value_;
}
bool nonnegative_int::operator!=(nonnegative_int const &other) const {
  return this->value_ != other.value_;
}
bool nonnegative_int::operator>=(nonnegative_int const &other) const {
  return this->value_ >= other.value_;
}

bool nonnegative_int::operator<(int const &other) const {
  return this->value_ < other;
}
bool nonnegative_int::operator==(int const &other) const {
  return this->value_ == other;
}
bool nonnegative_int::operator>(int const &other) const {
  return this->value_ > other;
}
bool nonnegative_int::operator<=(int const &other) const {
  return this->value_ <= other;
}
bool nonnegative_int::operator!=(int const &other) const {
  return this->value_ != other;
}
bool nonnegative_int::operator>=(int const &other) const {
  return this->value_ >= other;
}

bool operator<(int const &lhs, nonnegative_int const &rhs) {
  return lhs < rhs.value_;
}
bool operator==(int const &lhs, nonnegative_int const &rhs) {
  return lhs == rhs.value_;
}
bool operator>(int const &lhs, nonnegative_int const &rhs) {
  return lhs > rhs.value_;
}
bool operator<=(int const &lhs, nonnegative_int const &rhs) {
  return lhs <= rhs.value_;
}
bool operator!=(int const &lhs, nonnegative_int const &rhs) {
  return lhs != rhs.value_;
}
bool operator>=(int const &lhs, nonnegative_int const &rhs) {
  return lhs >= rhs.value_;
}

nonnegative_int nonnegative_int::operator+(nonnegative_int const &other) const {
  return nonnegative_int{this->value_ + other.value_};
}

std::ostream &operator<<(std::ostream &os, nonnegative_int const &n) {
  os << n.value_;
  return os;
}

int nonnegative_int::get_value() const {
  return this->value_;
}

int format_as(nonnegative_int const &x) {
  return x.get_value();
}
} // namespace FlexFlow

namespace nlohmann {
::FlexFlow::nonnegative_int
    adl_serializer<::FlexFlow::nonnegative_int>::from_json(json const &j) {
  return ::FlexFlow::nonnegative_int{j.template get<int>()};
}

void adl_serializer<::FlexFlow::nonnegative_int>::to_json(
    json &j, ::FlexFlow::nonnegative_int t) {
  j = t.get_value();
}
} // namespace nlohmann

namespace std {
std::size_t hash<::FlexFlow::nonnegative_int>::operator()(
    FlexFlow::nonnegative_int const &n) const noexcept {
  return std::hash<int>{}(n.get_value());
}
} // namespace std
