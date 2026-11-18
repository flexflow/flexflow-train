#include "import-pt2/pt2_archive.h"
#include <algorithm>
#include <fmt/format.h>
#include <fstream>
#include <optional>
#include <stdexcept>
#include <vector>

namespace FlexFlow {

namespace {

std::string join(std::vector<std::string> const &v) {
  std::string result;
  for (size_t i = 0; i < v.size(); i++) {
    result += (i == 0 ? "" : ", ") + v.at(i);
  }
  return result;
}

constexpr uint32_t EOCD_SIGNATURE = 0x06054b50;
constexpr uint32_t ZIP64_EOCD_LOCATOR_SIGNATURE = 0x07064b50;
constexpr uint32_t ZIP64_EOCD_SIGNATURE = 0x06064b50;
constexpr uint32_t CENTRAL_DIR_HEADER_SIGNATURE = 0x02014b50;
constexpr uint32_t LOCAL_HEADER_SIGNATURE = 0x04034b50;
constexpr uint16_t ZIP64_EXTRA_FIELD_ID = 0x0001;
constexpr size_t EOCD_SIZE = 22;
constexpr size_t ZIP64_EOCD_LOCATOR_SIZE = 20;
constexpr size_t ZIP64_EOCD_SIZE = 56;
constexpr size_t CENTRAL_DIR_HEADER_SIZE = 46;
constexpr size_t LOCAL_HEADER_SIZE = 30;
constexpr size_t MAX_ZIP_COMMENT_SIZE = 0xFFFF;

[[noreturn]] void fail(std::string const &path, std::string const &msg) {
  throw std::runtime_error(fmt::format("{}: {}", path, msg));
}

uint64_t read_le(std::string const &buf, size_t offset, size_t num_bytes) {
  if (offset + num_bytes > buf.size()) {
    throw std::out_of_range("zip record truncated");
  }
  uint64_t result = 0;
  for (size_t i = 0; i < num_bytes; i++) {
    result |= static_cast<uint64_t>(static_cast<uint8_t>(buf[offset + i]))
              << (8 * i);
  }
  return result;
}

uint16_t read_u16(std::string const &buf, size_t offset) {
  return static_cast<uint16_t>(read_le(buf, offset, 2));
}

uint32_t read_u32(std::string const &buf, size_t offset) {
  return static_cast<uint32_t>(read_le(buf, offset, 4));
}

uint64_t read_u64(std::string const &buf, size_t offset) {
  return read_le(buf, offset, 8);
}

std::ifstream open_binary(std::string const &path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) {
    fail(path, "could not open file");
  }
  return f;
}

std::string read_range(std::ifstream &f,
                       std::string const &path,
                       uint64_t offset,
                       uint64_t size) {
  std::string buf(size, '\0');
  f.seekg(static_cast<std::streamoff>(offset));
  f.read(buf.data(), static_cast<std::streamsize>(size));
  if (static_cast<uint64_t>(f.gcount()) != size) {
    fail(path,
         fmt::format("unexpected end of file reading {} bytes at offset {}",
                     size,
                     offset));
  }
  return buf;
}

} // namespace

std::map<std::string, ZipEntryLocation>
    read_zip_central_directory(std::string const &path) {
  std::ifstream f = open_binary(path);
  f.seekg(0, std::ios::end);
  uint64_t file_size = static_cast<uint64_t>(f.tellg());
  if (file_size < EOCD_SIZE) {
    fail(path, "not a zip file (too small)");
  }

  // The end-of-central-directory record is at the end of the file, followed
  // only by an optional comment of at most 64 KiB.
  uint64_t tail_size =
      std::min<uint64_t>(file_size, EOCD_SIZE + MAX_ZIP_COMMENT_SIZE);
  uint64_t tail_offset = file_size - tail_size;
  std::string tail = read_range(f, path, tail_offset, tail_size);

  std::optional<size_t> eocd_pos;
  for (size_t i = tail.size() - EOCD_SIZE + 1; i-- > 0;) {
    if (read_u32(tail, i) == EOCD_SIGNATURE) {
      eocd_pos = i;
      break;
    }
  }
  if (!eocd_pos.has_value()) {
    fail(path, "not a zip file (no end of central directory record)");
  }

  uint64_t num_entries = read_u16(tail, *eocd_pos + 10);
  uint64_t central_dir_size = read_u32(tail, *eocd_pos + 12);
  uint64_t central_dir_offset = read_u32(tail, *eocd_pos + 16);

  // zip64: the locator sits immediately before the classic record.
  uint64_t eocd_abs = tail_offset + *eocd_pos;
  if (eocd_abs >= ZIP64_EOCD_LOCATOR_SIZE) {
    std::string locator = read_range(
        f, path, eocd_abs - ZIP64_EOCD_LOCATOR_SIZE, ZIP64_EOCD_LOCATOR_SIZE);
    if (read_u32(locator, 0) == ZIP64_EOCD_LOCATOR_SIGNATURE) {
      uint64_t zip64_eocd_offset = read_u64(locator, 8);
      std::string zip64_eocd =
          read_range(f, path, zip64_eocd_offset, ZIP64_EOCD_SIZE);
      if (read_u32(zip64_eocd, 0) != ZIP64_EOCD_SIGNATURE) {
        fail(path, "bad zip64 end of central directory record");
      }
      num_entries = read_u64(zip64_eocd, 32);
      central_dir_size = read_u64(zip64_eocd, 40);
      central_dir_offset = read_u64(zip64_eocd, 48);
    }
  }

  std::string cd = read_range(f, path, central_dir_offset, central_dir_size);

  std::map<std::string, ZipEntryLocation> entries;
  size_t pos = 0;
  for (uint64_t i = 0; i < num_entries; i++) {
    if (read_u32(cd, pos) != CENTRAL_DIR_HEADER_SIGNATURE) {
      fail(path, fmt::format("bad central directory header for entry {}", i));
    }
    ZipEntryLocation loc;
    loc.compression_method = read_u16(cd, pos + 10);
    loc.compressed_size = read_u32(cd, pos + 20);
    loc.uncompressed_size = read_u32(cd, pos + 24);
    uint16_t name_len = read_u16(cd, pos + 28);
    uint16_t extra_len = read_u16(cd, pos + 30);
    uint16_t comment_len = read_u16(cd, pos + 32);
    loc.local_header_offset = read_u32(cd, pos + 42);
    std::string name = cd.substr(pos + CENTRAL_DIR_HEADER_SIZE, name_len);

    // zip64 extended information: only the fields that overflowed are
    // present, in this fixed order.
    size_t extra_pos = pos + CENTRAL_DIR_HEADER_SIZE + name_len;
    size_t extra_end = extra_pos + extra_len;
    while (extra_pos + 4 <= extra_end) {
      uint16_t id = read_u16(cd, extra_pos);
      uint16_t size = read_u16(cd, extra_pos + 2);
      if (id == ZIP64_EXTRA_FIELD_ID) {
        size_t p = extra_pos + 4;
        if (loc.uncompressed_size == 0xFFFFFFFF) {
          loc.uncompressed_size = read_u64(cd, p);
          p += 8;
        }
        if (loc.compressed_size == 0xFFFFFFFF) {
          loc.compressed_size = read_u64(cd, p);
          p += 8;
        }
        if (loc.local_header_offset == 0xFFFFFFFF) {
          loc.local_header_offset = read_u64(cd, p);
          p += 8;
        }
      }
      extra_pos += 4 + size;
    }

    entries.emplace(name, loc);
    pos += CENTRAL_DIR_HEADER_SIZE + name_len + extra_len + comment_len;
  }
  return entries;
}

std::string read_stored_zip_entry(std::string const &path,
                                  std::string const &entry_name,
                                  ZipEntryLocation const &location) {
  if (location.compression_method != 0) {
    fail(path,
         fmt::format("entry {} is compressed (method {}); only stored "
                     "(uncompressed) entries are supported",
                     entry_name,
                     location.compression_method));
  }
  std::ifstream f = open_binary(path);
  std::string header =
      read_range(f, path, location.local_header_offset, LOCAL_HEADER_SIZE);
  if (read_u32(header, 0) != LOCAL_HEADER_SIGNATURE) {
    fail(path, fmt::format("bad local header for entry {}", entry_name));
  }
  uint16_t name_len = read_u16(header, 26);
  uint16_t extra_len = read_u16(header, 28);
  uint64_t data_offset =
      location.local_header_offset + LOCAL_HEADER_SIZE + name_len + extra_len;
  return read_range(f, path, data_offset, location.uncompressed_size);
}

nlohmann::json read_pt2_model_json(std::string const &pt2_path,
                                   std::string const &model_name) {
  std::map<std::string, ZipEntryLocation> entries =
      read_zip_central_directory(pt2_path);

  auto strip_root = [](std::string const &name) -> std::optional<std::string> {
    size_t slash = name.find('/');
    if (slash == std::string::npos) {
      return std::nullopt;
    }
    return name.substr(slash + 1);
  };

  std::string wanted = fmt::format("models/{}.json", model_name);
  std::vector<std::string> matches;
  std::vector<std::string> available_models;
  for (auto const &[name, loc] : entries) {
    std::optional<std::string> rel = strip_root(name);
    if (!rel.has_value()) {
      continue;
    }
    if (*rel == wanted) {
      matches.push_back(name);
    } else if (rel->rfind("models/", 0) == 0 &&
               rel->find('/', 7) == std::string::npos) {
      available_models.push_back(*rel);
    }
    if (*rel == "archive_format") {
      std::string format = read_stored_zip_entry(pt2_path, name, loc);
      if (format != "pt2") {
        fail(pt2_path,
             fmt::format("archive_format is \"{}\", expected \"pt2\"", format));
      }
    }
  }

  if (matches.empty()) {
    fail(pt2_path,
         fmt::format("no {} in archive (found: [{}]). Is this a "
                     "torch.export.save() file?",
                     wanted,
                     join(available_models)));
  }
  if (matches.size() > 1) {
    fail(pt2_path,
         fmt::format("multiple {} entries: [{}]", wanted, join(matches)));
  }

  std::string contents =
      read_stored_zip_entry(pt2_path, matches.at(0), entries.at(matches.at(0)));
  try {
    return nlohmann::json::parse(contents);
  } catch (nlohmann::json::parse_error const &e) {
    fail(pt2_path,
         fmt::format("could not parse {}: {}", matches.at(0), e.what()));
  }
}

} // namespace FlexFlow
