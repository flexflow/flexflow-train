#include "utils/nonnegative_int/nonnegative_int.h"
#include "utils/exception.h"

namespace FlexFlow {

nonnegative_int::nonnegative_int(int value) {
  if (value < 0) {
    throw std::invalid_argument(
        "Value of nonnegative_int type must be nonnegative.");
  }
  this->value_ = value;
}

nonnegative_int::nonnegative_int(size_t value) {
  if (value > std::numeric_limits<int>::max()) {
    throw std::invalid_argument(fmt::format(
        "Input {} to nonnegative_int(size_t) is out-of-bounds for int", value));
  }
  this->value_ = static_cast<int>(value);
  assert(this->value_ >= 0);
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

nonnegative_int &nonnegative_int::operator++() {
  this->value_++;
  return *this;
}

nonnegative_int nonnegative_int::operator++(int) {
  nonnegative_int result = *this;
  this->value_++;
  return result;
}

nonnegative_int &nonnegative_int::operator+=(nonnegative_int const &other) {
  this->value_ += other.value_;
  return *this;
}

nonnegative_int nonnegative_int::operator-(nonnegative_int const &other) const {
  return nonnegative_int{this->value_ - other.value_};
}

nonnegative_int &nonnegative_int::operator-=(nonnegative_int const &other) {
  *this = nonnegative_int{this->value_ - other.value_};
  return *this;
}

nonnegative_int nonnegative_int::operator*(nonnegative_int const &other) const {
  return nonnegative_int{this->value_ * other.value_};
}

nonnegative_int &nonnegative_int::operator*=(nonnegative_int const &other) {
  this->value_ *= other.value_;
  return *this;
}

nonnegative_int nonnegative_int::operator/(nonnegative_int const &other) const {
  return nonnegative_int{this->value_ / other.value_};
}

nonnegative_int &nonnegative_int::operator/=(nonnegative_int const &other) {
  this->value_ /= other.value_;
  return *this;
}

nonnegative_int nonnegative_int::operator%(nonnegative_int const &other) const {
  return nonnegative_int{this->value_ % other.value_};
}

nonnegative_int &nonnegative_int::operator%=(nonnegative_int const &other) {
  this->value_ %= other.value_;
  return *this;
}

std::ostream &operator<<(std::ostream &os, nonnegative_int const &n) {
  os << n.value_;
  return os;
}

int nonnegative_int::unwrap_nonnegative() const {
  return this->value_;
}

int format_as(nonnegative_int const &x) {
  return x.unwrap_nonnegative();
}

nonnegative_int operator""_n(unsigned long long int x) {
  if (x >
      static_cast<unsigned long long int>(std::numeric_limits<int>::max())) {
    throw mk_runtime_error(
        fmt::format("Value too large to wrap as nonnegative_int: {}", x));
  }

  return nonnegative_int{static_cast<int>(x)};
}

} // namespace FlexFlow

namespace nlohmann {
::FlexFlow::nonnegative_int
    adl_serializer<::FlexFlow::nonnegative_int>::from_json(json const &j) {
  return ::FlexFlow::nonnegative_int{j.template get<int>()};
}

void adl_serializer<::FlexFlow::nonnegative_int>::to_json(
    json &j, ::FlexFlow::nonnegative_int t) {
  j = t.unwrap_nonnegative();
}
} // namespace nlohmann

namespace rc {
Gen<::FlexFlow::nonnegative_int>
    Arbitrary<::FlexFlow::nonnegative_int>::arbitrary() {
  return gen::construct<::FlexFlow::nonnegative_int>(gen::nonNegative<int>());
}
} // namespace rc

namespace std {
std::size_t hash<::FlexFlow::nonnegative_int>::operator()(
    FlexFlow::nonnegative_int const &n) const noexcept {
  return std::hash<int>{}(n.unwrap_nonnegative());
}
} // namespace std
