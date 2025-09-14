#ifndef _FLEXFLOW_LIB_COMPILER_INCLUDE_COMPILER_COST_ESTIMATOR_COMMUNICATION_EDGE_H
#define _FLEXFLOW_LIB_COMPILER_INCLUDE_COMPILER_COST_ESTIMATOR_COMMUNICATION_EDGE_H

#include "pcg/machine_space_coordinate.dtg.h"

namespace FlexFlow {

struct CommunicationEdge {
  CommunicationEdge() = delete;

  CommunicationEdge(MachineSpaceCoordinate const &src,
                    MachineSpaceCoordinate const &dst);

  bool operator==(CommunicationEdge const &) const;
  bool operator!=(CommunicationEdge const &) const;

  bool operator<(CommunicationEdge const &) const;
  bool operator>(CommunicationEdge const &) const;
  bool operator<=(CommunicationEdge const &) const;
  bool operator>=(CommunicationEdge const &) const;

  MachineSpaceCoordinate const &get_src() const;
  MachineSpaceCoordinate const &get_dst() const;
private:
  MachineSpaceCoordinate src;
  MachineSpaceCoordinate dst;
private:
  std::tuple<
    decltype(src) const &,
    decltype(dst) const &
  > tie() const;

  friend struct ::std::hash<CommunicationEdge>;
};

std::string format_as(CommunicationEdge const &);
std::ostream &operator<<(std::ostream &, CommunicationEdge const &);

} // namespace FlexFlow

namespace std {

template <>
struct hash<::FlexFlow::CommunicationEdge> {
  size_t operator()(::FlexFlow::CommunicationEdge const &) const;
};

}

#endif
