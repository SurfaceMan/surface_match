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

    BoundingBox()
        : min(Eigen::Vector3f::Zero())
        , max(Eigen::Vector3f::Zero()) {
    }

    BoundingBox(Eigen::Vector3f min_, Eigen::Vector3f max_)
        : min(min_)
        , max(max_) {
    }

    BoundingBox &operator=(const BoundingBox &rhs) {
        if (this != &rhs) {
            this->min = rhs.min;
            this->max = rhs.max;
        }

        return *this;
    }

    Eigen::Vector3f size() const {
        return max - min;
    }

    float diameter() const {
        return size().norm();
    }
};

struct PointCloud {
public:
    std::vector<Eigen::Vector3f> point;
    std::vector<Eigen::Vector3f> normal;
    BoundingBox                  box;
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

    BoxGrid()
        : step(0)
        , xBins(0)
        , yBins(0)
        , zBins(0) {
    }

    int grid2Index(const Eigen::Vector3i &index_)const {
        return index[ index_.x() ][ index_.y() ][ index_.z() ];
    }

    Eigen::Vector3i index2Grid(int index)const {
        return grid[ index ];
    }
};

struct Pose {
public:
    float           numVotes;
    Eigen::Affine3f pose;

    Eigen::AngleAxisf  r;
    Eigen::Quaternionf q;

    Pose(float votes)
        : numVotes(votes){};

    void updatePose(const Eigen::Matrix4f &newPose) {
        pose = newPose;

        auto rMatrix = pose.rotation();
        r            = rMatrix;
        q            = rMatrix;
    }

    void updatePoseT(const Eigen::Vector3f &t) {
        pose.translation() = t;
    }

    void updatePoseQuat(const Eigen::Quaternionf &q_) {
        q             = q_;
        r             = q.matrix();
        pose.linear() = q.matrix();
    }
};

} // namespace ppf