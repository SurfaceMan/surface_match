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
#include <KDTreeVectorOfVectorsAdaptor.h>
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

    explicit Pose(float votes);

    void updatePose(const Eigen::Matrix4f &newPose);
    void updatePoseT(const Eigen::Vector3f &t);
    void updatePoseQuat(const Eigen::Quaternionf &q);
};

/**
 * @brief additional parameter for training model
 *
 */
struct TrainParam {
public:
    /**
     * @brief Set the discretion distance of the point pair distance
     * relative to the object's diameter
     */
    float featDistanceStepRel;

    /**
     * @brief Set the discretion of the point pair orientation as the number
     * of subdivisions of the angle
     */
    int featAngleResolution;

    /**
     * @brief Set the sampling distance for the pose refinement relative to the object's diameter
     *
     */
    float poseRefRelSamplingDistance;

    explicit TrainParam(float featDistanceStepRel = 0.04f, int featAngleResolution = 30,
                        float poseRefRelSamplingDistance = 0.01f);
};

/**
 * @brief additional parameter for matching scene
 *
 */
struct MatchParam {
public:
    /**
     * @brief The neighbors count threshold of key point. less than this won't compute. The value is
     * set relative to the point count of the model.
     *
     */
    float voteThresholdFraction;

    /**
     * @brief Sets the maximum number of matches that are returned
     *
     */
    int numMatches;

    /**
     * @brief The minimum distance between the centers of the axis-aligned bounding boxes of two
     * matches. The value is set relative to the diameter of the object.
     *
     */
    float maxOverlapDistRel;

    /**
     * @brief This parameter has the same effect as the parameter 'maxOverlapDistRel'. Note that
     * in contrast to 'maxOverlapDistRel', the value for 'maxOverlapDistAbs' is set as an
     * absolute value
     *
     */
    float maxOverlapDistAbs;

    /**
     * @brief Enables or disables the sparse pose refinement
     *
     */
    bool sparsePoseRefinement;

    /**
     * @brief Enables or disables the dense pose refinement.
     *
     */
    bool densePoseRefinement;

    /**
     * @brief Number of iterations for the dense pose refinement.
     *
     */
    int poseRefNumSteps;

    /**
     * @brief Set the distance threshold for dense pose refinement relative to the diameter of the
     * surface model. Only scene points that are closer to the object than this distance are used
     * for the optimization. Scene points further away are ignored.
     *
     */
    float poseRefDistThresholdRel;

    /**
     * @brief This parameter has the same effect as the parameter 'poseRefDistThresholdRel'. Note
     * that in contrast to 'poseRefDistThresholdRel', the value for 'poseRefDistThresholdAbs' is set
     * as an absolute value
     *
     */
    float poseRefDistThresholdAbs;

    /**
     * @brief Set the distance threshold for scoring relative to the diameter of the surface model
     *
     */
    float poseRefScoringDistRel;

    /**
     * @brief Set the distance threshold for scoring. Only scene points that are closer to the
     * object than this distance are considered to be 'on the model' when computing the score after
     * the pose refinement. All other scene points are considered not to be on the model. The value
     * should correspond to the amount of noise on the coordinates of the scene points.
     *
     */
    float poseRefScoringDistAbs;

    explicit MatchParam(float voteThresholdFraction = 0.2f, int numMatches = 1,
                        float maxOverlapDistRel = 0.5f, float maxOverlapDistAbs = 0,
                        bool sparsePoseRefinement = true, bool densePoseRefinement = true,
                        int poseRefNumSteps = 5, float poseRefDistThresholdRel = 0.1f,
                        float poseRefDistThresholdAbs = 0, float poseRefScoringDistRel = 0.01f,
                        float poseRefScoringDistAbs = 0);
};

using KDTree = KDTreeVectorOfVectorsAdaptor<std::vector<Eigen::Vector3f>, float>;

} // namespace ppf