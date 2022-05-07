#pragma once

#include <Eigen/Geometry>
#include <kdtree.h>
#include <ppf.h>

namespace ppf {
struct Feature {
public:
    int   refInd;
    float alphaAngle;
    float voteValue;

    Feature()
        : refInd(0)
        , alphaAngle(0)
        , voteValue(0) {
    }

    Feature(int refInd_, float alphaAngle_, float voteValue_)
        : refInd(refInd_)
        , alphaAngle(alphaAngle_)
        , voteValue(voteValue_) {
    }
};

struct Candidate {
public:
    Candidate(float vote_, int refId_, int angleId_)
        : vote(vote_)
        , refId(refId_)
        , angleId(angleId_) {
    }

    float vote = 0;
    int   refId;
    int   angleId;
};

struct Detector::IMPL {
public:
    // model
    float      samplingDistanceRel;
    TrainParam param;

    PointCloud sampledModel;
    PointCloud reSampledModel;

    std::unordered_map<uint32_t, std::vector<Feature>> hashTable;
};

struct Pose {
public:
    Eigen::Matrix4f    pose;
    Eigen::AngleAxisf  r;
    Eigen::Quaternionf q;
    float              numVotes;

    explicit Pose(float votes);

    void updatePose(const Eigen::Matrix4f &newPose);
    void updatePoseT(const Eigen::Vector3f &t);
    void updatePoseQuat(const Eigen::Quaternionf &q);

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

using KDTree = KDTreeVectorOfVectorsAdaptor<std::vector<Eigen::Vector3f>, float>;

} // namespace ppf