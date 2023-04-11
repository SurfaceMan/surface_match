#pragma once

#include <Eigen/Geometry>
#include <gtl/phmap.hpp>
#include <ppf.h>
#include <xsimd/xsimd.hpp>

#define _USE_MATH_DEFINES
#include <math.h>

namespace ppf {

const float M_2PI = static_cast<float>(2 * M_PI);

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

    Eigen::Matrix<T, 3, 1, 0> operator[](size_t idx) const {
        return Eigen::Matrix<T, 3, 1, 0>{x[ idx ], y[ idx ], z[ idx ]};
    }

    void push_back(T x_, T y_, T z_) {
        x.push_back(x_);
        y.push_back(y_);
        z.push_back(z_);
    }

    void push_back(Eigen::Matrix<T, 3, 1, 0> p) {
        x.push_back(p.x());
        y.push_back(p.y());
        z.push_back(p.z());
    }

    void resize(size_t count) {
        x.resize(count);
        y.resize(count);
        z.resize(count);
    }

    void resize(size_t count, T x_, T y_, T z_) {
        x.resize(count, x_);
        y.resize(count, y_);
        z.resize(count, z_);
    }

    void resize(size_t count, Eigen::Matrix<T, 3, 1, 0> p) {
        x.resize(count, p.x());
        y.resize(count, p.y());
        z.resize(count, p.z());
    }

    void set(size_t idx, T x_, T y_, T z_) {
        x[ idx ] = x_;
        y[ idx ] = y_;
        z[ idx ] = z_;
    }

    void set(size_t idx, Eigen::Matrix<T, 3, 1, 0> p) {
        x[ idx ] = p.x();
        y[ idx ] = p.y();
        z[ idx ] = p.z();
    }

    [[nodiscard]] bool empty() const {
        return x.empty();
    }
};

using Vector3I = Vector3<int>;
using Vector3F = Vector3<float>;
using Vector3C = Vector3<unsigned char>;

using VectorF = std::vector<float, xsimd::aligned_allocator<float>>;
using VectorI = std::vector<uint32_t, xsimd::aligned_allocator<uint32_t>>;

struct BoundingBox {
    Eigen::Vector3f _min;
    Eigen::Vector3f _max;
    Eigen::Vector3f _size;
    Eigen::Vector3f _center;
    float           _diameter = 0;

public:
    BoundingBox();
    BoundingBox(Eigen::Vector3f min, Eigen::Vector3f max);
    BoundingBox &operator=(const BoundingBox &rhs);

    [[nodiscard]] Eigen::Vector3f min() const;
    [[nodiscard]] Eigen::Vector3f max() const;
    [[nodiscard]] Eigen::Vector3f size() const;
    [[nodiscard]] Eigen::Vector3f center() const;
    [[nodiscard]] float           diameter() const;
};

struct PointCloud {
public:
    Vector3F        point;
    Vector3F        normal;
    Vector3I        face;
    Vector3C        color;
    BoundingBox     box;
    Eigen::Vector3f viewPoint = Eigen::Vector3f(NAN, NAN, NAN);

    PointCloud();
    ~PointCloud();

    [[nodiscard]] bool        hasNormal() const;
    [[nodiscard]] std::size_t size() const;
    [[nodiscard]] bool        empty() const;
};

struct Feature {
public:
    VectorI refInd;
    VectorF alphaAngle;

    void push_back(uint32_t index, float angle);
};

struct Candidate {
public:
    float vote = 0;
    int   refId;
    int   angleId;

    Candidate(float vote, int refId, int angleId);
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
    void updatePoseR(const Eigen::Quaternionf &q);

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

} // namespace ppf
