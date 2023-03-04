#pragma once

#include <Eigen/Geometry>
#include <gtl/phmap.hpp>
#include <kdtree.h>
#include <ppf.h>
#include <xsimd/xsimd.hpp>

namespace ppf {

using vectorF = std::vector<float, xsimd::aligned_allocator<float>>;
using vectorI = std::vector<uint32_t, xsimd::aligned_allocator<uint32_t>>;

/*struct Feature {
public:
    int   refInd;
    float alphaAngle;
    float voteValue;

    Feature()
        : refInd(0)
        , alphaAngle(0) {
    }

    Feature(int refInd_, float alphaAngle_)
        : refInd(refInd_)
        , alphaAngle(alphaAngle_) {
    }
};*/

struct Feature {
public:
    vectorI refInd;
    vectorF alphaAngle;

    void push_back(uint32_t index, float angle) {
        refInd.push_back(index);
        alphaAngle.push_back(angle);
    }
    // Feature &operator=(Feature &&rhs) noexcept;
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

    gtl::flat_hash_map<uint32_t, Feature> hashTable;
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