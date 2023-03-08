#pragma once

#include <Eigen/Geometry>
#include <gtl/phmap.hpp>
#include <kdtree.h>
#include <ppf.h>
#include <xsimd/xsimd.hpp>

namespace ppf {

using vectorF = std::vector<float, xsimd::aligned_allocator<float>>;
using vectorI = std::vector<uint32_t, xsimd::aligned_allocator<uint32_t>>;

struct Feature {
public:
    vectorI refInd;
    vectorF alphaAngle;

    void push_back(uint32_t index, float angle) {
        refInd.push_back(index);
        alphaAngle.push_back(angle);
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

template <typename T> struct Vector3 {
public:
    using VectorT = std::vector<T, xsimd::aligned_allocator<T>>;
    VectorT x;
    VectorT y;
    VectorT z;

    Vector3() = default;
    explicit Vector3(size_t size) {
        resize(size);
    }

    [[nodiscard]] size_t size() const {
        return x.size();
    }

    Eigen::Matrix<float, 3, 1> operator[](size_t idx) const {
        return Eigen::Matrix<T, 3, 1>{x[ idx ], y[ idx ], z[ idx ]};
    }

    void push_back(const Eigen::Matrix<T, 3, 1> &element) {
        x.push_back(element.x());
        y.push_back(element.y());
        z.push_back(element.z());
    }

    void resize(size_t count) {
        x.resize(count);
        y.resize(count);
        z.resize(count);
    }

    void resize(size_t count, Eigen::Matrix<T, 3, 1> value) {
        x.resize(count, value(0));
        y.resize(count, value(1));
        z.resize(count, value(2));
    }

    [[nodiscard]] bool empty() const {
        return x.empty();
    }
};

using Vector3I = Vector3<int>;
using Vector3F = Vector3<float>;

using KDTree = KDTreeVectorOfVectorsAdaptor<std::vector<Eigen::Vector3f>, float>;
} // namespace ppf