#pragma once

#include <icp.h>
#include <kdtree.h>
#include <privateType.h>
#include <type.h>

#include <set>

namespace ppf {

/**
 * @brief sampleMesh sample mesh by radius
 * @param pc Input mesh
 * @param radius Sampled point distance
 * @return Sampled point cloud
 */
PointCloud sampleMesh(const ppf::PointCloud &pc, float radius);

/**
 * @brief removeNan Check point cloud points or normals(if exists) is nan
 * @param pc Input point cloud
 * @param checkNormal Check normal or not
 * @return Indices of element which is not nan
 */
VectorI removeNan(const ppf::PointCloud &pc, bool checkNormal = false);

/**
 * @brief extraIndices Select part of point cloud by indices
 * @param pc Input point cloud
 * @param indices Input indices
 * @return Selected point cloud
 */
PointCloud extraIndices(const ppf::PointCloud &pc, const VectorI &indices);

/**
 * @brief normalizeNormal Normalize normal to range 0~1
 * @param pc    Input/Output Point Cloud
 * @param invert Invert normal direction
 */
void normalizeNormal(ppf::PointCloud &pc, bool invert = false);

/**
 * @brief computeBoundingBox Compute aligned bounding box
 * @param pc Input point cloud
 * @param validIndices Only compute these indices
 * @return
 */
BoundingBox computeBoundingBox(const ppf::PointCloud &pc, const VectorI &validIndices = {});

/**
 * @brief samplePointCloud Sample point cloud by radius
 * @param tree  KD-tree
 * @param sampleStep Sample radius
 * @param indicesOfIndices Indices of keeped indices in the order which build kdtree
 * @return Indices of element which kepp
 */
VectorI samplePointCloud(const KDTree &tree, float sampleStep, VectorI *indicesOfIndices = nullptr);

/**
 * @brief estimateNormal Estimate normal of point cloud
 * @param pc    Input point cloud
 * @param indices   Which points to estiamte normal
 * @param kdtree    Kdtree to find neighbor
 * @param k The count of neighbors to compute normal
 * @param smooth    Smooth normal by neighbor's normal
 * @param invert    Invert normal direction
 */
void estimateNormal(ppf::PointCloud &pc, const VectorI &indices, const KDTree &kdtree, int k = 10,
                    bool smooth = true, bool invert = false);

/**
 * @brief estimateNormalMLS Estimate normal of point cloud with MLS(Moving Least Squares)
 * @param pc    Input point cloud
 * @param indices   Which points to estiamte normal
 * @param kdtree    Kdtree to find neighbor
 * @param radius    The radius of neighbors to compute normal
 * @param order
 * @param invert    Invert normal direction
 */
// void estimateNormalMLS(ppf::PointCloud &pc, const VectorI &indices, const KDTree &kdtree,
//                        float radius, int order, bool invert = false);

/**
 * @brief transformRT
 * @param p
 * @param n
 * @return
 */
Eigen::Matrix4f transformRT(const Eigen::Vector3f &p, const Eigen::Vector3f &n);

/**
 * @brief xRotMat
 * @param angle
 * @return
 */
inline Eigen::Matrix4f xRotMat(float angle) {
    Eigen::Matrix4f T;
    T << 1, 0, 0, 0, 0, cos(angle), -sin(angle), 0, 0, sin(angle), cos(angle), 0, 0, 0, 0, 1;

    return T;
}

void computeVote(VectorI &accumulator, const VectorI &id, const VectorF &angle, VectorI &idxAngle,
                 float alphaScene, float maxId, int accElementSize);

bool nms(Pose &target, const VectorI &accumulator, float voteThreshold, int refNum, int angleNum,
         int accElementSize, int maxAngleIndex, const PointCloud &modelSampled,
         const Eigen::Matrix4f &rt);

bool icp(const Pose &p, float &score, Eigen::Matrix4f &pose, const MatchParam &param,
         const ICP &sparseIcp, const ICP &denseIcp, KDTree &sceneKdtree, const PointCloud &model,
         const PointCloud &rModel, const PointCloud &scene, const VectorI &indicesOfSampleScene,
         const VectorI &indicesOfSampleScene2, float minScore, float poseRefScoringDist);

/**
 * @brief clusterPose   Cluster pose by distance/angle threshold
 * @param poseList  Input poses to cluster
 * @param distanceThreshold
 * @param angleThreshold
 * @return  Pose Clusters
 */
std::vector<std::vector<Pose>> clusterPose(const std::vector<Pose> &poseList,
                                           float distanceThreshold, float angleThreshold);

/**
 * @brief cluster by overlap
 * @param poseList
 * @param pos
 * @param threshold
 * @return std::vector<Pose>
 */
std::vector<Pose> clusterPose2(std::vector<Pose> &poseList, Eigen::Vector3f &pos, float threshold);

/**
 * @brief sortPoses Sort pose by score(vote num)
 * @param poseList
 * @return Sorted pose
 */
std::vector<Pose> sortPoses(std::vector<Pose> poseList);

/**
 * @brief avgClusters   Pose cluster to a average pose
 * @param clusters  Input Pose Cluster
 * @return Average Poses
 */
std::vector<Pose> avgClusters(const std::vector<std::vector<Pose>> &clusters);

/**
 * @brief findClosestPoint
 * @param kdtree
 * @param srcPC
 * @param indices
 * @param distances
 */
void findClosestPoint(const KDTree &kdtree, const PointCloud &srcPC, VectorI &indices,
                      std::vector<float> &distances);

/**
 * @brief inliner   Find count of source point  which close to target within threshold
 * @param srcPC Source point cloud
 * @param kdtree    Target kd-tree
 * @param inlineDist    Distance threshold
 * @return  Count of points fit threshold
 */
int inliner(const PointCloud &srcPC, const KDTree &kdtree, float inlineDist);

/**
 * @brief transformPointCloud
 * @param pc
 * @param pose
 * @param useNormal
 * @return
 */
PointCloud transformPointCloud(const ppf::PointCloud &pc, const Eigen::Matrix4f &pose,
                               bool useNormal = true);

VectorI computePPF(const Eigen::Vector3f &p1, const Eigen::Vector3f &n1, const VectorF &p2x,
                   const VectorF &p2y, const VectorF &p2z, const VectorF &n2x, const VectorF &n2y,
                   const VectorF &n2z, float angleStep, float distStep);

VectorF computeAlpha(Eigen::Matrix4f &rt, const VectorF &p2x, const VectorF &p2y,
                     const VectorF &p2z);

} // namespace ppf
