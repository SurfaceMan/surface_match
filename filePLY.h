#pragma once

#include <type.h>

namespace ppf {
bool readPLY(const std::string &filename, ppf::PointCloud &mesh);
bool writePLY(const std::string &filename, const ppf::PointCloud &mesh, bool write_ascii = false);
} // namespace ppf