#include "utils/dot/dot_html_from_json.h"
#include <libassert/assert.hpp>

namespace FlexFlow {

DotHtmlTable dot_html_table_from_json(nlohmann::json const &j) {
  auto mk_table_from_rows =
      [](std::vector<DotHtmlTableRow> const &rows) -> DotHtmlTable {
    return DotHtmlTable{
        /*border=*/0_n,
        /*cellborder=*/1_n,
        /*cellspacing=*/0_n,
        /*rows=*/rows,
    };
  };

  auto mk_singleton_table = [&](std::string const &s) -> DotHtmlTable {
    return mk_table_from_rows({
        DotHtmlTableRow{
            /*cells=*/{
                DotHtmlTableCell{
                    /*content=*/DotHtmlTableCellContents{
                        s,
                    },
                    /*port=*/std::nullopt,
                    /*colspan=*/std::nullopt,
                },
            },
        },
    });
  };

  auto mk_singleton_row = [&](nlohmann::json const &v) -> DotHtmlTableRow {
    return DotHtmlTableRow{
        /*cells=*/{
            dot_html_cell_from_json(v),
        },
    };
  };

  auto mk_kv_row = [&](std::string const &k,
                       nlohmann::json const &v) -> DotHtmlTableRow {
    return DotHtmlTableRow{
        /*cells=*/{
            DotHtmlTableCell{
                /*content=*/DotHtmlTableCellContents{
                    k,
                },
                /*port=*/std::nullopt,
                /*colspan=*/std::nullopt,
            },
            dot_html_cell_from_json(v),
        },
    };
  };

  switch (j.type()) {
    case nlohmann::json::value_t::null:
      return mk_singleton_table("(none)");
    case nlohmann::json::value_t::boolean:
      return mk_singleton_table(fmt::to_string(j.get<bool>()));
    case nlohmann::json::value_t::string:
      return mk_singleton_table(j.get<std::string>());
    case nlohmann::json::value_t::number_integer:
      return mk_singleton_table(fmt::to_string(j.get<int>()));
    case nlohmann::json::value_t::number_unsigned:
      return mk_singleton_table(fmt::to_string(j.get<unsigned int>()));
    case nlohmann::json::value_t::number_float:
      return mk_singleton_table(fmt::to_string(j.get<float>()));
    case nlohmann::json::value_t::object: {
      std::vector<DotHtmlTableRow> rows;
      for (auto const &[k, v] : j.items()) {
        rows.push_back(mk_kv_row(k, v));
      }
      return mk_table_from_rows(rows);
    }
    case nlohmann::json::value_t::array: {
      std::vector<DotHtmlTableRow> rows;
      for (auto const &v : j) {
        rows.push_back(mk_singleton_row(v));
      }
      return mk_table_from_rows(rows);
    }

    case nlohmann::json::value_t::binary:
    case nlohmann::json::value_t::discarded:
    default:
      PANIC("Unhandled value_t", j.type());
  }
}

DotHtmlTableCell dot_html_cell_from_json(nlohmann::json const &j) {

  auto mk_cell_from_string = [](std::string const &s) -> DotHtmlTableCell {
    return DotHtmlTableCell{
        DotHtmlTableCellContents{
            s,
        },
        /*port=*/std::nullopt,
        /*colspan=*/std::nullopt,
    };
  };

  auto mk_cell_from_table = [](DotHtmlTable const &t) -> DotHtmlTableCell {
    return DotHtmlTableCell{
        DotHtmlTableCellContents{
            t,
        },
        /*port=*/std::nullopt,
        /*colspan=*/std::nullopt,
    };
  };

  switch (j.type()) {
    case nlohmann::json::value_t::null:
      return mk_cell_from_string("(none)");
    case nlohmann::json::value_t::boolean:
      return mk_cell_from_string(fmt::to_string(j.get<bool>()));
    case nlohmann::json::value_t::string:
      return mk_cell_from_string(j.get<std::string>());
    case nlohmann::json::value_t::number_integer:
      return mk_cell_from_string(fmt::to_string(j.get<int>()));
    case nlohmann::json::value_t::number_unsigned:
      return mk_cell_from_string(fmt::to_string(j.get<unsigned int>()));
    case nlohmann::json::value_t::number_float:
      return mk_cell_from_string(fmt::to_string(j.get<float>()));
    case nlohmann::json::value_t::object:
    case nlohmann::json::value_t::array:
      return mk_cell_from_table(dot_html_table_from_json(j));
    case nlohmann::json::value_t::binary:
    case nlohmann::json::value_t::discarded:
    default:
      PANIC("Unhandled value_t", j.type());
  }
}

} // namespace FlexFlow
