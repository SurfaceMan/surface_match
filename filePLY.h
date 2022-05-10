#pragma once

#include <apiExport.h>
#include <type.h>

namespace ppf {
bool API_PUBLIC readPLY(const std::string &filename, ppf::PointCloud &mesh);
bool API_PUBLIC writePLY(const std::string &filename, const ppf::PointCloud &mesh,
                         bool write_ascii = false);
} // namespace ppf