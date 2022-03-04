/**
 * @file type.h
 * @author y.qiu (y.qiu@pixoel.com)
 * @brief
 * @version 0.1
 * @date 2022-02-16
 *
 * Copyright (c) 2021 Pixoel Technologies Co.ltd.
 *
 */

#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <map>
#include <tuple>
#include <vector>

namespace ppf {

struct BoundingBox {
public:
    Eigen::Vector3f min;
    Eigen::Vector3f max;

    BoundingBox();
    BoundingBox(Eigen::Vector3f min, Eigen::Vector3f max);
    BoundingBox &operator=(const BoundingBox &rhs);

    Eigen::Vector3f size() const;
    Eigen::Vector3f center() const;
    float           diameter() const;
};

struct PointCloud {
public:
    std::vector<Eigen::Vector3f> point;
    std::vector<Eigen::Vector3f> normal;
    BoundingBox                  box;

    bool hasNormal() const;
};

struct BoxGrid {
public:
    static const int INVALID = -1;

    std::vector<Eigen::Vector3i>               grid;  // PointCloud--->grid
    std::vector<std::vector<std::vector<int>>> index; // grid--->PointCloud

    float step; // size of box
    int   xBins;
    int   yBins;
    int   zBins;

    BoxGrid();

    int             grid2Index(const Eigen::Vector3i &grid) const;
    Eigen::Vector3i index2Grid(int index) const;
};

struct Pose {
public:
    float           numVotes;
    Eigen::Affine3f pose;

    Eigen::AngleAxisf  r;
    Eigen::Quaternionf q;

    Pose(float votes);

    void updatePose(const Eigen::Matrix4f &newPose);
    void updatePoseT(const Eigen::Vector3f &t);
    void updatePoseQuat(const Eigen::Quaternionf &q);
};

} // namespace ppf