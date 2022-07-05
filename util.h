#pragma once

#include <apiExport.h>
#include <type.h>

namespace ppf {

API_PUBLIC PointCloud sampleMesh(const ppf::PointCloud &pc, float radius);

API_PUBLIC std::vector<int> removeNan(const ppf::PointCloud &pc, bool checkNormal = false);

API_PUBLIC PointCloud extraIndices(const ppf::PointCloud          &pc,
                                   const std::vector<std::size_t> &indices);

API_PUBLIC void normalizeNormal(ppf::PointCloud &pc, bool invert = false);

API_PUBLIC BoundingBox computeBoundingBox(const ppf::PointCloud  &pc,
                                          const std::vector<int> &validIndices = {});

API_PUBLIC PointCloud transformPointCloud(const ppf::PointCloud &pc, const Eigen::Matrix4f &pose,
                                          bool useNormal = true);

} // namespace ppf