#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_DOT_DOT_HTML_FROM_JSON_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_DOT_DOT_HTML_FROM_JSON_H

#include "utils/dot/dot_html_table.dtg.h"
#include <nlohmann/json.hpp>

namespace FlexFlow {

DotHtmlTable dot_html_table_from_json(nlohmann::json const &);
DotHtmlTableCell dot_html_cell_from_json(nlohmann::json const &);

} // namespace FlexFlow

#endif
