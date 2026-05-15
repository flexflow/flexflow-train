#include "utils/record_formatter.h"
#include "utils/archetypes/ordered_value_type.h"
#include "utils/archetypes/value_type.h"

namespace FlexFlow {

RecordFormatter::RecordFormatter(Orientation orientation,
                                 std::vector<std::string> const &pieces)
    : orientation(orientation), pieces(pieces) {}

RecordFormatter mk_empty_record(Orientation o) {
  return RecordFormatter{o, std::vector<std::string>{}};
}

RecordFormatter &operator<<(RecordFormatter &r, std::string const &tok) {
  r.pieces.push_back(tok);

  return r;
}

RecordFormatter &operator<<(RecordFormatter &r, int tok) {
  std::ostringstream oss;
  oss << tok;

  r << oss;

  return r;
}

RecordFormatter &operator<<(RecordFormatter &r, float tok) {
  std::ostringstream oss;
  oss << std::scientific;
  oss << tok;

  r << oss;

  return r;
}

RecordFormatter &operator<<(RecordFormatter &r, RecordFormatter const &sub_r) {
  std::ostringstream oss;

  if (r.orientation == sub_r.orientation) {
    oss << "{ " << sub_r << " }";
  } else {
    oss << sub_r;
  }
  r << oss.str();

  return r;
}

RecordFormatter &operator<<(RecordFormatter &r, std::ostringstream &oss) {
  r << oss.str();

  return r;
}

std::ostream &operator<<(std::ostream &s, RecordFormatter const &r) {
  s << "{ ";
  for (size_t i = 0; i < r.pieces.size(); i++) {
    s << r.pieces[i];
    if (i + 1 < r.pieces.size()) {
      s << " | ";
    }
  }
  s << " }";

  return s;
}

template <>
RecordFormatter mk_kv_record(std::string const &k, RecordFormatter const &v) {
  RecordFormatter rr = mk_empty_record(Orientation::HORIZONTAL);
  rr << k << v;
  return rr;
}

} // namespace FlexFlow

namespace FlexFlow {

using T = value_type<0>;

template RecordFormatter mk_kv_record(std::string const &, T const &);

template RecordFormatter mk_kv_record(std::string const &,
                                      std::optional<T> const &);

using K = ordered_value_type<0>;
using V = value_type<0>;

template RecordFormatter mk_record_for_map(std::unordered_map<K, V> const &);

} // namespace FlexFlow
