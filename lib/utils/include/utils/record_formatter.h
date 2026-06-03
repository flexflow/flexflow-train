#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_RECORD_FORMATTER_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_RECORD_FORMATTER_H

#include "utils/containers/keys.h"
#include "utils/containers/sorted.h"
#include "utils/orientation.dtg.h"
#include <optional>
#include <sstream>
#include <vector>

namespace FlexFlow {

/**
 * \brief Helper interface for generating
 * <a href="https://graphviz.org/doc/info/shapes.html#record">DOT/graphviz records</a>.
 *
 * \note This is very old code and should not be emulated stylistically.
 *
 * \see \ref DotFile
 * \see \ref mk_empty_record
 */
class RecordFormatter {
public:
  RecordFormatter() = delete;
  explicit RecordFormatter(Orientation, std::vector<std::string> const &pieces);

  friend RecordFormatter &operator<<(RecordFormatter &r,
                                     std::string const &tok);
  friend RecordFormatter &operator<<(RecordFormatter &r, int tok);
  friend RecordFormatter &operator<<(RecordFormatter &r, float tok);
  friend RecordFormatter &operator<<(RecordFormatter &r,
                                     RecordFormatter const &sub_r);
  friend RecordFormatter &operator<<(RecordFormatter &r,
                                     std::ostringstream &oss);
  friend std::ostream &operator<<(std::ostream &s, RecordFormatter const &r);

public:
  Orientation orientation;
  std::vector<std::string> pieces;
};

RecordFormatter mk_empty_record(Orientation);

template <typename T>
RecordFormatter mk_kv_record(std::string const &k, T const &v) {
  RecordFormatter rr = mk_empty_record(Orientation::HORIZONTAL);
  rr << k << fmt::to_string(v);
  return rr;
}

template <>
RecordFormatter mk_kv_record(std::string const &, RecordFormatter const &);

template <typename T>
RecordFormatter mk_kv_record(std::string const &k, std::optional<T> const &v) {
  if (v.has_value()) {
    return mk_kv_record(k, v.value());
  } else {
    RecordFormatter rr = mk_empty_record(Orientation::HORIZONTAL);
    rr << k << "(none)";
    return rr;
  }
}

template <typename K, typename V>
RecordFormatter mk_record_for_map(std::unordered_map<K, V> const &m) {
  RecordFormatter result = mk_empty_record(Orientation::VERTICAL);

  for (K const &k : sorted(keys(m))) {
    result << mk_kv_record(fmt::to_string(k), m.at(k));
  }

  return result;
}

} // namespace FlexFlow

#endif
