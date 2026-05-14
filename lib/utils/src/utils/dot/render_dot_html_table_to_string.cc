#include "utils/dot/render_dot_html_table_to_string.h"
#include "utils/containers/flatmap.h"
#include "utils/containers/transform.h"
#include "utils/join_strings.h"

namespace FlexFlow {

static std::string escape_html_string(std::string const &s) {
  auto escape_dot_char = [](char c) -> std::string {
    switch (c) {
      case '<':
        return std::string{"&lt;"};
      case '>':
        return std::string{"&gt;"};
      default:
        return std::string{c};
    }
  };

  return flatmap(s, escape_dot_char);
}

static std::string render_dot_html_cell_contents_to_string(
    DotHtmlTableCellContents const &cell_contents) {
  if (cell_contents.is_simple()) {
    return escape_html_string(cell_contents.require_simple());
  } else {
    return render_dot_html_table_to_string(cell_contents.require_nested());
  }
}

static std::string
    render_dot_html_cell_to_string(DotHtmlTableCell const &cell) {
  std::ostringstream oss;

  oss << "<TD";
  if (cell.port.has_value()) {
    oss << " PORT=\"" << cell.port.value() << "\"";
  }
  if (cell.colspan.has_value()) {
    oss << " COLSPAN=\"" << cell.colspan.value() << "\"";
  }
  oss << ">" << render_dot_html_cell_contents_to_string(cell.content)
      << "</TD>";

  return oss.str();
}

static std::string render_dot_html_row_to_string(DotHtmlTableRow const &row) {
  return fmt::format(
      "<TR>{}</TR>",
      join_strings(transform(row.cells, render_dot_html_cell_to_string),
                   std::string{"\n"}));
}

std::string render_dot_html_table_to_string(DotHtmlTable const &table) {
  return fmt::format(
      "<TABLE BORDER=\"{}\" CELLBORDER=\"{}\" CELLSPACING=\"{}\">{}</TABLE>",
      table.border,
      table.cellborder,
      table.cellspacing,
      join_strings(transform(table.rows, render_dot_html_row_to_string),
                   std::string{"\n"}));
}

} // namespace FlexFlow
