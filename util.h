/**
 * @file util.h
 * @author y.qiu (y.qiu@pixoel.com)
 * @brief
 * @version 0.1
 * @date 2022-02-16
 *
 * Copyright (c) 2021 Pixoel Technologies Co.ltd.
 *
 */

#pragma once

#include "type.h"

namespace ppf {

/**
 * @brief load PointCloud from ply file
 *
 * @param filename
 * @param pc
 * @return true success
 * @return false failed
 */
bool loadPLY(const std::string &filename, ppf::PointCloud &pc);

/**
 * @brief save PointCloud to ply file
 *
 * @param filename
 * @param pc
 */
void savePLY(const std::string &filename, const ppf::PointCloud &pc);

/**
 * @brief sample PointCloud with distance step(sampleStep)
 *
 * @param pc
 * @param sampleStep
 * @param grid
 * @return ppf::PointCloud
 */
PointCloud samplePointCloud(const ppf::PointCloud &pc, float sampleStep, BoxGrid *grid = nullptr);

BoundingBox computeBoundingBox(const ppf::PointCloud &pc);

PointCloud transformPointCloud(const ppf::PointCloud &pc, const Eigen::Matrix4f &pose);

std::vector<Eigen::Vector3f> estimateNormal(const ppf::PointCloud &pc);

std::vector<Eigen::Vector3f> estimateNormal(const ppf::PointCloud &pc, const ppf::PointCloud &ref);

Eigen::Vector4f computePPF(const Eigen::Vector3f &p1, const Eigen::Vector3f &p2,
                           const Eigen::Vector3f &n1, const Eigen::Vector3f &n2);

uint32_t hashPPF(const Eigen::Vector4f &ppfValue, float angleRadians, float distanceStep);

void transformRT(const Eigen::Vector3f &p, const Eigen::Vector3f &n, Eigen::Matrix3f &R,
                 Eigen::Vector3f &t);

float computeAlpha(const Eigen::Vector3f &p1, const Eigen::Vector3f &p2, const Eigen::Vector3f &n1);

inline Eigen::Matrix4f XRotMat(float angle) {
    Eigen::Matrix4f T;
    T << 1, 0, 0, 0, 0, cos(angle), -sin(angle), 0, 0, sin(angle), cos(angle), 0, 0, 0, 0, 1;

    return T;
}

std::vector<std::vector<Pose>> clusterPose(const std::vector<Pose> &poseList,
                                           float distanceThreshold, float angleThreshold);

/**
 * @brief cluster by overlap
 *
 * @param poseList
 * @param pos
 * @param threshold
 * @return std::vector<Pose>
 */
std::vector<Pose> clusterPose2(std::vector<Pose> &poseList, Eigen::Vector3f &pos, float threshold);

std::vector<Pose> sortPoses(std::vector<Pose> poseList);

std::vector<Pose> avgClusters(const std::vector<std::vector<Pose>> &clusters);

PointCloud loadText(const std::string &filename);

void saveText(const std::string &filename, const PointCloud &pc);
} // namespace ppf