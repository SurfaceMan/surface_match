/**
 * @file filePLY.h
 * @author y.qiu (y.qiu@pixoel.com)
 * @brief
 * @version 0.1
 * @date 2022-04-14
 *
 * @copyright Copyright (c) 2021 Pixoel Technologies Co.ltd.
 *
 */

#pragma once

#include <type.h>

namespace ppf {
bool readPLY(const std::string &filename, ppf::PointCloud &mesh);
bool writePLY(const std::string &filename, const ppf::PointCloud &mesh, bool write_ascii = false);
} // namespace ppf