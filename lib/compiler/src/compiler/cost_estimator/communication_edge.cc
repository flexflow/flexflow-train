#include "compiler/cost_estimator/communication_edge.h"
#include "utils/hash-utils.h"
#include "utils/hash/tuple.h"
#include <libassert/assert.hpp>

namespace FlexFlow {

CommunicationEdge::CommunicationEdge(MachineSpaceCoordinate const &src,
                                     MachineSpaceCoordinate const &dst)
    : src(src), dst(dst) {
  ASSERT(src != dst);
}

bool CommunicationEdge::operator==(CommunicationEdge const &other) const {
  return this->tie() == other.tie();
}

bool CommunicationEdge::operator!=(CommunicationEdge const &other) const {
  return this->tie() != other.tie();
}

bool CommunicationEdge::operator<(CommunicationEdge const &other) const {
  return this->tie() < other.tie();
}

bool CommunicationEdge::operator>(CommunicationEdge const &other) const {
  return this->tie() > other.tie();
}

bool CommunicationEdge::operator<=(CommunicationEdge const &other) const {
  return this->tie() <= other.tie();
}

bool CommunicationEdge::operator>=(CommunicationEdge const &other) const {
  return this->tie() >= other.tie();
}

MachineSpaceCoordinate const &CommunicationEdge::get_src() const {
  return this->src;
}

MachineSpaceCoordinate const &CommunicationEdge::get_dst() const {
  return this->dst;
}

std::tuple<MachineSpaceCoordinate const &, MachineSpaceCoordinate const &>
    CommunicationEdge::tie() const {
  return std::tie(this->src, this->dst);
}

std::string format_as(CommunicationEdge const &e) {
  return fmt::format(
      "<CommunicationEdge src={} dst={}>", e.get_src(), e.get_dst());
}

std::ostream &operator<<(std::ostream &s, CommunicationEdge const &e) {
  return (s << fmt::to_string(e));
}

} // namespace FlexFlow

namespace std {

size_t hash<::FlexFlow::CommunicationEdge>::operator()(
    ::FlexFlow::CommunicationEdge const &e) const {
  return get_std_hash(e.tie());
}

} // namespace std
