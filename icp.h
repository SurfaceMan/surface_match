/**
 * @file icp.h
 * @author y.qiu (y.qiu@pixoel.com)
 * @brief
 * @version 0.1
 * @date 2022-02-17
 *
 * Copyright (c) 2021 Pixoel Technologies Co.ltd.
 *
 */

#pragma once

#include <type.h>

#include <memory>

namespace ppf {
class ICP {
public:
    ICP(const int iterations, const float tolerence = 0.05f, const float rejectionScale = 2.5f);
    ~ICP();

    int registerModelToScene(const PointCloud &srcPC, const PointCloud &dstPC, float &residual,
                             Eigen::Matrix4f &pose);

    int registerModelToScene(const PointCloud &srcPC, const PointCloud &dstPC,
                             std::vector<float> &residual, std::vector<Eigen::Matrix4f> &pose);

private:
    struct IMPL;
    std::unique_ptr<IMPL> impl_;
};
} // namespace ppf