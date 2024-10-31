#include "utils/nonnegative_int/nonnegative_int.h"

namespace FlexFlow {

nonnegative_int::nonnegative_int(int const &value) {
  if (value < 0) {
    throw std::invalid_argument(
        "Value of nonnegative_int type must be nonnegative.");
  }
  this->value_ = value;
}
nonnegative_int::nonnegative_int(int &&value) {
  if (std::move(value) < 0) {
    throw std::invalid_argument(
        "Value of nonnegative_int type must be nonnegative.");
  }
  this->value_ = std::move(value);
}

void swap(nonnegative_int &a, nonnegative_int &b) noexcept {
  using std::swap;
  swap(static_cast<int &>(a), static_cast<int &>(b));
}

nonnegative_int::operator int() const noexcept {
  return this->value_;
}
nonnegative_int::operator int &() noexcept {
  return this->value_;
}
nonnegative_int::operator int const &() const noexcept {
  return this->value_;
}

bool nonnegative_int::operator<(nonnegative_int const &other) {
  return this->value_ < other.value_;
}
bool nonnegative_int::operator==(nonnegative_int const &other) {
  return this->value_ == other.value_;
}
bool nonnegative_int::operator>(nonnegative_int const &other) {
  return this->value_ > other.value_;
}
bool nonnegative_int::operator<=(nonnegative_int const &other) {
  return this->value_ <= other.value_;
}
bool nonnegative_int::operator!=(nonnegative_int const &other) {
  return this->value_ != other.value_;
}
bool nonnegative_int::operator>=(nonnegative_int const &other) {
  return this->value_ >= other.value_;
}

nonnegative_int &nonnegative_int::operator+=(nonnegative_int const &other) {
  static_cast<int &>(*this) += static_cast<int const &>(other);
  return *this;
}
nonnegative_int &nonnegative_int::operator+=(int const &other) {
  static_cast<int &>(*this) += other;
  return *this;
}
nonnegative_int &nonnegative_int::operator++() {
  return *this += 1;
}
nonnegative_int nonnegative_int::operator++(int) {
  nonnegative_int tmp = *this;
  ++*this;
  return tmp;
}
nonnegative_int nonnegative_int::operator+(nonnegative_int const &other) {
  return nonnegative_int(this->value_ + other.value_);
}
nonnegative_int nonnegative_int::operator+(int const &other) {
  return nonnegative_int(this->value_ + other);
}
nonnegative_int operator+(int const &lhs, nonnegative_int const &rhs) {
  return nonnegative_int(lhs + rhs.value_);
}

nonnegative_int &nonnegative_int::operator-=(nonnegative_int const &other) {
  if (*this < other) {
    throw std::out_of_range("Invalid subtraction");
  }
  static_cast<int &>(*this) -= static_cast<int const &>(other);
  return *this;
}
nonnegative_int &nonnegative_int::operator-=(int const &other) {
  if (*this < other) {
    throw std::out_of_range("Invalid subtraction");
  }
  static_cast<int &>(*this) -= other;
  return *this;
}
nonnegative_int &nonnegative_int::operator--() {
  return *this -= 1;
}
nonnegative_int nonnegative_int::operator--(int) {
  nonnegative_int tmp = *this;
  --*this;
  return tmp;
}
nonnegative_int nonnegative_int::operator-(nonnegative_int const &other) {
  if (*this < other) {
    throw std::out_of_range("Invalid subtraction");
  }
  return nonnegative_int(this->value_ - other.value_);
}
nonnegative_int nonnegative_int::operator-(int const &other) {
  if (*this < other) {
    throw std::out_of_range("Invalid subtraction");
  }
  return nonnegative_int(this->value_ - other);
}
nonnegative_int operator-(int const &lhs, nonnegative_int const &rhs) {
  if (lhs < rhs) {
    throw std::out_of_range("Invalid subtraction");
  }
  return nonnegative_int(lhs - rhs.value_);
}

nonnegative_int &nonnegative_int::operator*=(nonnegative_int const &other) {
  static_cast<int &>(*this) *= static_cast<int const &>(other);
  return *this;
}
nonnegative_int &nonnegative_int::operator*=(int const &other) {
  static_cast<int &>(*this) *= other;
  return *this;
}
nonnegative_int nonnegative_int::operator*(nonnegative_int const &other) {
  return nonnegative_int(this->value_ * other.value_);
}
nonnegative_int nonnegative_int::operator*(int const &other) {
  return nonnegative_int(this->value_ * other);
}
nonnegative_int operator*(int const &lhs, nonnegative_int const &rhs) {
  return nonnegative_int(lhs * rhs.value_);
}

nonnegative_int &nonnegative_int::operator/=(nonnegative_int const &other) {
  if (other == 0) {
    throw std::invalid_argument("Cannot divide by zero");
  }
  static_cast<int &>(*this) /= static_cast<int const &>(other);
  return *this;
}
nonnegative_int &nonnegative_int::operator/=(int const &other) {
  if (other == 0) {
    throw std::invalid_argument("Cannot divide by zero");
  }
  static_cast<int &>(*this) /= other;
  return *this;
}
nonnegative_int nonnegative_int::operator/(nonnegative_int const &other) {
  return (other != 0 ? nonnegative_int(this->value_ / other.value_)
                     : throw std::invalid_argument("Cannot divide by zero"));
}
nonnegative_int nonnegative_int::operator/(int const &other) {
  return (other != 0 ? nonnegative_int(this->value_ / other)
                     : throw std::invalid_argument("Cannot divide by zero"));
}
nonnegative_int operator/(int const &lhs, nonnegative_int const &rhs) {
  return (rhs != 0 ? nonnegative_int(lhs / rhs.value_)
                   : throw std::invalid_argument("Cannot divide by zero"));
}

std::ostream &operator<<(std::ostream &os, nonnegative_int const &n) {
  os << n.value_;
  return os;
}

int nonnegative_int::get_value() const {
  return this->value_;
}
} // namespace FlexFlow

namespace std {} // namespace std
