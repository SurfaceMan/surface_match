#pragma once

#include <apiExport.h>
#include <string>
#include <type.h>

namespace ppf {
bool API_PUBLIC readPLY(const std::string &filename, PointCloud_t*mesh);
bool API_PUBLIC writePLY(const std::string &filename, const PointCloud_t mesh,
                         bool write_ascii = false);
} // namespace ppf
