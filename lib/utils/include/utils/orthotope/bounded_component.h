#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_ORTHOTOPE_BOUNDED_COMPONENT_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_ORTHOTOPE_BOUNDED_COMPONENT_H

#include "utils/fmt/json.h"
#include "utils/nonnegative_int/nonnegative_int.h"
#include "utils/positive_int/positive_int.h"

namespace FlexFlow {

struct BoundedComponent {
public:
  BoundedComponent() = delete;
  explicit BoundedComponent(nonnegative_int component, positive_int bound);

  [[nodiscard]] bool operator==(BoundedComponent const &) const;
  [[nodiscard]] bool operator!=(BoundedComponent const &) const;

  [[nodiscard]] bool operator<(BoundedComponent const &) const;
  [[nodiscard]] bool operator>(BoundedComponent const &) const;
  [[nodiscard]] bool operator<=(BoundedComponent const &) const;
  [[nodiscard]] bool operator>=(BoundedComponent const &) const;

  [[nodiscard]] nonnegative_int const &get_component() const;
  [[nodiscard]] positive_int const &get_bound() const;

private:
  nonnegative_int component;
  positive_int bound;

private:
  [[nodiscard]] std::tuple<decltype(component) const &, decltype(bound) const &>
      tie() const;

  friend ::std::hash<::FlexFlow::BoundedComponent>;
};

nlohmann::json format_as(BoundedComponent const &);
std::ostream &operator<<(std::ostream &, BoundedComponent const &);

BoundedComponent trivial_bounded_component();

} // namespace FlexFlow

namespace std {

template <>
struct hash<::FlexFlow::BoundedComponent> {
  size_t operator()(::FlexFlow::BoundedComponent const &) const;
};

} // namespace std

namespace nlohmann {

template <>
struct adl_serializer<::FlexFlow::BoundedComponent> {
  static ::FlexFlow::BoundedComponent from_json(json const &);
  static void to_json(json &, ::FlexFlow::BoundedComponent const &);
};

} // namespace nlohmann

namespace rc {

template <>
struct Arbitrary<::FlexFlow::BoundedComponent> {
  static Gen<::FlexFlow::BoundedComponent> arbitrary();
};

} // namespace rc

#endif
