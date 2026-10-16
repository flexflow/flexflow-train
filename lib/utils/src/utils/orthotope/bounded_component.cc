#include "utils/orthotope/bounded_component.h"
#include "utils/hash-utils.h"
#include "utils/hash/tuple.h"
#include <libassert/assert.hpp>

namespace FlexFlow {

BoundedComponent::BoundedComponent(nonnegative_int c, positive_int b)
    : component(c), bound(b) {
  ASSERT(this->component < this->bound);
}

bool BoundedComponent::operator==(BoundedComponent const &other) const {
  return this->tie() == other.tie();
}

bool BoundedComponent::operator!=(BoundedComponent const &other) const {
  return this->tie() != other.tie();
}

bool BoundedComponent::operator<(BoundedComponent const &other) const {
  return this->tie() < other.tie();
}

bool BoundedComponent::operator>(BoundedComponent const &other) const {
  return this->tie() > other.tie();
}

bool BoundedComponent::operator<=(BoundedComponent const &other) const {
  return this->tie() <= other.tie();
}

bool BoundedComponent::operator>=(BoundedComponent const &other) const {
  return this->tie() >= other.tie();
}

nonnegative_int const &BoundedComponent::get_component() const {
  return this->component;
}

positive_int const &BoundedComponent::get_bound() const {
  return this->bound;
}

std::tuple<nonnegative_int const &, positive_int const &>
    BoundedComponent::tie() const {
  return std::tie(this->component, this->bound);
}

nlohmann::json format_as(BoundedComponent const &x) {
  nlohmann::json j = x;
  return j;
}

std::ostream &operator<<(std::ostream &s, BoundedComponent const &x) {
  return (s << fmt::to_string(x));
}

BoundedComponent trivial_bounded_component() {
  return BoundedComponent{
      /*component=*/0_n,
      /*bound=*/1_p,
  };
}

} // namespace FlexFlow

namespace std {

size_t hash<::FlexFlow::BoundedComponent>::operator()(
    ::FlexFlow::BoundedComponent const &x) const {
  return ::FlexFlow::get_std_hash(x.tie());
}

} // namespace std

namespace nlohmann {

::FlexFlow::BoundedComponent
    adl_serializer<::FlexFlow::BoundedComponent>::from_json(json const &j) {

  return ::FlexFlow::BoundedComponent{
      /*component=*/j.at("component")
          .template get<::FlexFlow::nonnegative_int>(),
      /*bound=*/j.at("bound").template get<::FlexFlow::positive_int>(),
  };
}

void adl_serializer<::FlexFlow::BoundedComponent>::to_json(
    json &j, ::FlexFlow::BoundedComponent const &x) {
  j["__type"] = "BoundedComponent";
  j["component"] = x.get_component();
  j["bound"] = x.get_bound();
}

} // namespace nlohmann

namespace rc {

Gen<::FlexFlow::BoundedComponent>
    Arbitrary<::FlexFlow::BoundedComponent>::arbitrary() {
  using namespace FlexFlow;

  return gen::mapcat(
      gen::arbitrary<positive_int>(),
      [](positive_int bound) -> Gen<BoundedComponent> {
        Gen<positive_int> bound_gen = gen::just(bound);
        Gen<nonnegative_int> component_gen = gen::construct<nonnegative_int>(
            gen::inRange(0, bound.int_from_positive_int()));
        return gen::construct<BoundedComponent>(component_gen, bound_gen);
      });
}

} // namespace rc
