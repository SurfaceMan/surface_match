#pragma once

#include <Eigen/Geometry>
#include <apiExport.h>
#include <type.h>

namespace ppf {

API_DEMANGLED API_PUBLIC PointCloud_t PointCloud_New();

API_DEMANGLED API_PUBLIC void PointCloud_Delete(PointCloud_t *handle);

API_DEMANGLED API_PUBLIC void PointCloud_SetViewPoint(PointCloud_t handle, float x, float y,
                                                      float z);

} // namespace ppf
