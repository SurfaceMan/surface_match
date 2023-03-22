#pragma once

#include <apiExport.h>

namespace ppf {

struct PointCloud;

using PointCloud_t = PointCloud *;

struct MatchResult {
    PointCloud_t sampledScene;
    PointCloud_t keyPoint;
};

/**
 * @brief additional parameter for training model
 *
 */
struct API_PUBLIC TrainParam {
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

    /**
     * @brief Set the count of nearest neighbors to estimate normal
     *
     */
    int knnNormal;

    /**
     * @brief Enables or disables smooth normal by neighbors
     *
     */
    bool smoothNormal;

    explicit TrainParam(float featDistanceStepRel = 0.04f, int featAngleResolution = 30,
                        float poseRefRelSamplingDistance = 0.01f, int knnNormal = 10,
                        bool smoothNormal = true);
};

/**
 * @brief additional parameter for matching scene
 *
 */
struct API_PUBLIC MatchParam {
public:
    /**
     * @brief Sets the maximum number of matches that are returned
     *
     */
    int numMatches;

    /**
     * @brief Set the count of nearest neighbors to estimate normal
     *
     */
    int knnNormal;

    /**
     * @brief Enables or disables smooth normal by neighbors
     *
     */
    bool smoothNormal;

    /**
     * @brief Invert the orientation of the scene
     *
     */
    bool invertNormal;

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

    explicit MatchParam(int numMatches = 1, int knnNormal = 10, bool smoothNormal = true,
                        bool invertNormal = false, float maxOverlapDistRel = 0.5f,
                        float maxOverlapDistAbs = 0, bool sparsePoseRefinement = true,
                        bool densePoseRefinement = true, int poseRefNumSteps = 5,
                        float poseRefDistThresholdRel = 0.1f, float poseRefDistThresholdAbs = 0,
                        float poseRefScoringDistRel = 0.01f, float poseRefScoringDistAbs = 0);
};
} // namespace ppf
