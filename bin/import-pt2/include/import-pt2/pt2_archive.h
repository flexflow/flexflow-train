#ifndef _FLEXFLOW_BIN_IMPORT_PT2_INCLUDE_IMPORT_PT2_PT2_ARCHIVE_H
#define _FLEXFLOW_BIN_IMPORT_PT2_INCLUDE_IMPORT_PT2_PT2_ARCHIVE_H

#include <cstdint>
#include <map>
#include <nlohmann/json.hpp>
#include <string>

namespace FlexFlow {

/**
 * @brief Location of one entry inside a zip file.
 */
struct ZipEntryLocation {
  uint16_t compression_method;
  uint64_t compressed_size;
  uint64_t uncompressed_size;
  uint64_t local_header_offset;
};

/**
 * @brief Read the central directory of the zip file at \p path (zip64 aware).
 *
 * @return a map from entry name to its location.
 */
std::map<std::string, ZipEntryLocation>
    read_zip_central_directory(std::string const &path);

/**
 * @brief Read the contents of a stored (uncompressed) zip entry.
 *
 * PyTorch's archive writer stores every entry uncompressed, so no inflate
 * support is needed. Throws if the entry is compressed.
 */
std::string read_stored_zip_entry(std::string const &path,
                                  std::string const &entry_name,
                                  ZipEntryLocation const &location);

/**
 * @brief Read and parse <tt>models/{model_name}.json</tt> (the serialized
 * ExportedProgram) from a <tt>torch.export.save</tt> .pt2 archive.
 *
 * The archive's top-level directory is not fixed (PyTorch names it after the
 * file), so any single top-level directory is accepted.
 */
nlohmann::json read_pt2_model_json(std::string const &pt2_path,
                                   std::string const &model_name = "model");

} // namespace FlexFlow

#endif
