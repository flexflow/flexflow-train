#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_DOT_DOT_HTML_TABLE_CELL_CONTENTS_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_DOT_DOT_HTML_TABLE_CELL_CONTENTS_H

#include <memory>
#include <string>
#include <variant>

namespace FlexFlow {

struct DotHtmlTable;

struct DotHtmlTableCellContents {
public:
  DotHtmlTableCellContents() = delete;
  explicit DotHtmlTableCellContents(std::string const &);
  explicit DotHtmlTableCellContents(DotHtmlTable const &);

  DotHtmlTable const &require_nested() const;
  std::string const &require_simple() const;

  bool is_simple() const;
  bool is_nested() const;

  bool operator==(DotHtmlTableCellContents const &) const;
  bool operator!=(DotHtmlTableCellContents const &) const;

private:
  std::variant<std::string, std::shared_ptr<DotHtmlTable>> value;
};

std::string format_as(DotHtmlTableCellContents const &);
std::ostream &operator<<(std::ostream &, DotHtmlTableCellContents const &);

} // namespace FlexFlow

#endif
