#pragma once

#include <apiExport.h>
#include <string>
#include <type.h>

namespace ppf {
API_DEMANGLED API_PUBLIC bool PointCloud_ReadPLY(const char *filename, PointCloud_t *mesh);
API_DEMANGLED API_PUBLIC bool PointCloud_WritePLY(const char *filename, PointCloud_t mesh,
                                                  bool write_ascii = false);
} // namespace ppf
