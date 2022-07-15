#pragma once

#include <privateType.h>
#include <type.h>

namespace ppf {
std::vector<std::size_t> samplePointCloud(const KDTree &tree, float sampleStep,
                                          std::vector<int> *indicesOfIndices = nullptr);

void estimateNormal(ppf::PointCloud &pc, const std::vector<std::size_t> &indices,
                    const KDTree &kdtree, int k = 10, bool smooth = true, bool invert = false);

void estimateNormalMLS(ppf::PointCloud &pc, const std::vector<std::size_t> &indices,
                       const KDTree &kdtree, float radius, int order, bool invert = false);

Eigen::Matrix4f transformRT(const Eigen::Vector3f &p, const Eigen::Vector3f &n);

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

void findClosestPoint(const KDTree &kdtree, const PointCloud &srcPC, std::vector<int> &indices,
                      std::vector<float> &distances);

int inliner(const PointCloud &srcPC, const KDTree &kdtree, float inlineDist);

vectorI computePPF(const Eigen::Vector3f &p1, const Eigen::Vector3f &n1, const vectorF &p2x,
                   const vectorF &p2y, const vectorF &p2z, const vectorF &n2x, const vectorF &n2y,
                   const vectorF &n2z, float angleStep, float distStep);

vectorF computeAlpha(Eigen::Matrix4f &rt, const vectorF &p2x, const vectorF &p2y,
                     const vectorF &p2z);

} // namespace ppf