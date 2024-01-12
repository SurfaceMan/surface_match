#include <privateType.h>
#include <privateUtil.h>
#include <util.h>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <Eigen/StdVector>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <random>

#define _USE_MATH_DEFINES
#include <math.h>

namespace ppf {

PointCloud sampleMesh(const ppf::PointCloud &pc, float radius) {
    if (pc.face.empty())
        return {};

    // compute normal /area
    auto                         triangleSize = pc.face.size();
    std::vector<float>           areas(triangleSize);
    std::vector<Eigen::Vector3f> normals(triangleSize);
    float                        area = 0;
    for (int i = 0; i < triangleSize; i++) {
        auto &face = pc.face[ i ];
        auto &p0   = pc.point[ face[ 0 ] ];
        auto &p1   = pc.point[ face[ 1 ] ];
        auto &p2   = pc.point[ face[ 2 ] ];

        auto            u = p1 - p0;
        auto            v = p2 - p0;
        Eigen::Vector3f k = u.cross(v);

        areas[ i ]   = k.norm() / 2.f;
        normals[ i ] = k.normalized();
        area += areas[ i ];
    }

    // integral
    areas[ 0 ] = areas[ 0 ] / area;
    for (int i = 1; i < triangleSize; i++)
        areas[ i ] = areas[ i ] / area + areas[ i - 1 ];
    areas[ triangleSize - 1 ] = 1.f;
    // number of point
    int nPoints = area / (radius * radius);

    //
    std::random_device                    rd;
    int                                   seed = rd();
    std::mt19937                          mt(seed);
    std::uniform_real_distribution<float> dist(0.0, 1.0);
    PointCloud                            result;

    result.point.resize(nPoints);
    result.normal.resize(nPoints);
    size_t pointIdx = 0;
    for (int i = 0; i < triangleSize; i++) {
        size_t n = size_t(std::round(areas[ i ] * nPoints));
        while (pointIdx < n) {
            auto r1 = dist(mt);
            auto r2 = dist(mt);
            auto a  = 1. - std::sqrt(r1);
            auto b  = std::sqrt(r1) * (1. - r2);
            auto c  = std::sqrt(r1) * r2;

            auto &face                = pc.face[ i ];
            auto &p0                  = pc.point[ face[ 0 ] ];
            auto &p1                  = pc.point[ face[ 1 ] ];
            auto &p2                  = pc.point[ face[ 2 ] ];
            result.point[ pointIdx ]  = a * p0 + b * p1 + c * p2;
            result.normal[ pointIdx ] = normals[ i ];

            pointIdx++;
        }
    }

    result.box = pc.box;
    return result;
}

std::vector<int> removeNan(const ppf::PointCloud &pc, bool checkNormal) {
    auto              size = pc.size();
    std::vector<bool> keep(size, true);
    checkNormal = checkNormal && pc.hasNormal();

#pragma omp parallel for
    for (int i = 0; i < size; i++) {
        if (!pc.point[ i ].allFinite() || (checkNormal && !pc.normal[ i ].allFinite()))
            keep[ i ] = false;
    }

    std::vector<int> result;
    result.reserve(size);
    for (int i = 0; i < size; i++) {
        if (keep[ i ])
            result.push_back(i);
    }

    return result;
}

std::vector<std::size_t> samplePointCloud(const KDTree &tree, float sampleStep,
                                          std::vector<int> *indicesOfIndices) {

    auto                     size = tree.index->vAcc_.size();
    std::vector<bool>        keep(tree.m_data.size(), true);
    auto                     radius = sampleStep * sampleStep;
    std::vector<std::size_t> result;
    if (indicesOfIndices)
        indicesOfIndices->resize(size, nanoflann::INVALID_INDEX);

#pragma omp parallel for schedule(static)
    for (int i = 0; i < size; i++) {
        auto index = tree.index->vAcc_[ i ];
        if (index == nanoflann::INVALID_INDEX)
            continue;

        if (!keep[ index ])
            continue;
#pragma omp critical
        { result.push_back(index); }
        if (indicesOfIndices)
            (*indicesOfIndices)[ i ] = index;

        auto                                  &point = tree.m_data[ index ];
        std::vector<nanoflann::ResultItem<int, float>>     indices;
        nanoflann::RadiusResultSet<float, int> resultSet(radius, indices);
        tree.index->findNeighbors(resultSet, &point[ 0 ], nanoflann::SearchParameters(0, false));
        for (auto &[ idx, dist ] : indices)
            keep[ idx ] = false;
    }

    return result;
}

PointCloud extraIndices(const ppf::PointCloud &pc, const std::vector<std::size_t> &indices) {
    PointCloud result;
    bool       hasNormal = pc.hasNormal();

    result.point.resize(indices.size());
    if (hasNormal)
        result.normal.resize(indices.size());

#pragma omp parallel for
    for (int i = 0; i < indices.size(); i++) {
        result.point[ i ] = pc.point[ indices[ i ] ];
        if (hasNormal)
            result.normal[ i ] = pc.normal[ indices[ i ] ].normalized();
    }

    result.box = pc.box;
    if (result.box.diameter() == 0)
        result.box = computeBoundingBox(result);

    return result;
}

void normalizeNormal(ppf::PointCloud &pc, bool invert) {

    if (invert) {
#pragma omp parallel for
        for (int i = 0; i < pc.normal.size(); i++) {
            pc.normal[ i ] = -pc.normal[ i ].normalized();
        }
        return;
    }
#pragma omp parallel for
    for (int i = 0; i < pc.normal.size(); i++) {
        pc.normal[ i ].normalize();
    }
}

BoundingBox computeBoundingBox(const ppf::PointCloud &pc, const std::vector<int> &validIndices) {
    Eigen::Vector3f min = validIndices.empty() ? pc.point[ 0 ] : pc.point[ validIndices[ 0 ] ];
    Eigen::Vector3f max = min;

    // bounding box
    if (validIndices.empty()) {
#pragma omp parallel for
        for (int dim = 0; dim < 3; dim++) {
            for (auto &p : pc.point) {
                if (p[ dim ] > max[ dim ])
                    max[ dim ] = p[ dim ];
                else if (p[ dim ] < min[ dim ])
                    min[ dim ] = p[ dim ];
            }
        }
    } else {
#pragma omp parallel for
        for (int dim = 0; dim < 3; dim++) {
            for (auto idx : validIndices) {
                auto &p = pc.point[ idx ];
                if (p[ dim ] > max[ dim ])
                    max[ dim ] = p[ dim ];
                else if (p[ dim ] < min[ dim ])
                    min[ dim ] = p[ dim ];
            }
        }
    }

    return {min, max};
}

PointCloud transformPointCloud(const ppf::PointCloud &pc, const Eigen::Matrix4f &pose,
                               bool useNormal) {
    auto size     = pc.size();
    auto doNormal = pc.hasNormal() && useNormal;

    PointCloud result;
    result.point.resize(size);
    if (doNormal)
        result.normal.resize(size);

    auto r = pose.topLeftCorner(3, 3);
    auto t = pose.topRightCorner(3, 1);

#pragma omp parallel for
    for (int i = 0; i < size; i++) {
        result.point[ i ] = r * pc.point[ i ] + t;
        if (doNormal)
            result.normal[ i ] = r * pc.normal[ i ];
    }

    result.face = pc.face;
    result.box  = computeBoundingBox(result);
    return result;
}

void computeNormal(ppf::PointCloud &pc, int idx, const KDTree &tree, int k,
                   std::vector<int> *neighbour = nullptr) {
    auto &point  = pc.point[ idx ];
    auto &normal = pc.normal[ idx ];
    if (normal.allFinite())
        return;

    // neighbour
    std::vector<int>   indices(k);
    std::vector<float> dists(k);
    tree.index->knnSearch(&point[ 0 ], k, indices.data(), dists.data());
    if (indices.size() < 3)
        return;

    std::vector<Eigen::Vector3f> neighbours(indices.size());
    for (int j = 0; j < indices.size(); j++)
        neighbours[ j ] = tree.m_data[ indices[ j ] ];

    if (neighbour)
        (*neighbour) = indices;

    // pca
    Eigen::Map<const Eigen::Matrix3Xf> P(neighbours[ 0 ].data(), 3, neighbours.size());
    Eigen::Vector3f                    centroid = P.rowwise().mean();
    Eigen::Matrix3Xf                   centered = P.colwise() - centroid;
    Eigen::Matrix3f                    cov      = centered * centered.transpose();

    // eigvecs sorted in increasing order of eigvals
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eig;
    eig.computeDirect(cov);
    Eigen::Vector3f eval   = eig.eigenvalues();
    int             minInd = 0;
    eval.cwiseAbs().minCoeff(&minInd);
    normal = eig.eigenvectors().col(minInd); // is already normalized

    point = centroid;
}

void estimateNormal(ppf::PointCloud &pc, const std::vector<std::size_t> &indices,
                    const KDTree &tree, int k, bool smooth, bool invert) {
    if (!pc.hasNormal())
        pc.normal.resize(pc.point.size(), Eigen::Vector3f(NAN, NAN, NAN));
    if (k < 3)
        k = 3;

    auto size = indices.size();
#pragma omp parallel for
    for (int i = 0; i < size; i++) {
        auto  idx    = indices[ i ];
        auto &normal = pc.normal[ idx ];
        auto &point  = pc.point[ idx ];

        std::vector<int> neighbor;
        computeNormal(pc, idx, tree, k, &neighbor);

        if (neighbor.empty())
            continue;

        if (smooth) {
            for (auto index : neighbor)
                computeNormal(pc, index, tree, k);

            Eigen::Vector3f nSum(0, 0, 0);
            for (auto index : neighbor) {
                auto &n = pc.normal[ index ];
                if (n.dot(normal) > 0)
                    nSum += n;
                else
                    nSum -= n;
            }

            normal = nSum.normalized();
        }
    }

    std::function<bool(const float &)> check = [](const float &val) { return val < 0.f; };
    if (invert)
        check = [](const float &val) { return val > 0.f; };

    if (pc.viewPoint.allFinite()) {
#pragma omp parallel for
        for (int i = 0; i < size; i++) {
            auto  idx    = indices[ i ];
            auto &normal = pc.normal[ idx ];
            auto &point  = pc.point[ idx ];
            if (check(normal.dot(pc.viewPoint - point)))
                normal = -normal;
            continue;
        }
    } else {
        // normal's direction default toward z axis
#pragma omp parallel for
        for (int i = 0; i < size; i++) {
            auto  idx    = indices[ i ];
            auto &normal = pc.normal[ idx ];
            if (check(normal.dot(Eigen::Vector3f::UnitZ())))
                normal = -normal;
            continue;
        }
    }
}

void estimateNormalMLS(ppf::PointCloud &pc, const std::vector<std::size_t> &indices,
                       const KDTree &kdtree, float radius, int order, bool invert) {
    // Compute the number of coefficients
    int  nCoeff = (order + 1) * (order + 2) / 2;
    auto r2     = radius * radius;

    if (!pc.hasNormal())
        pc.normal.resize(pc.point.size(), Eigen::Vector3f(NAN, NAN, NAN));

    std::function<bool(const float &)> check = [](const float &val) { return val < 0.f; };
    if (invert)
        check = [](const float &val) { return val > 0.f; };

    auto size = indices.size();
#pragma omp parallel for
    for (int i = 0; i < size; i++) {
        auto  idx    = indices[ i ];
        auto &normal = pc.normal[ idx ];
        auto &point  = pc.point[ idx ];

        // neighbour
        std::vector<nanoflann::ResultItem<int, float>>     indices;
        nanoflann::RadiusResultSet<float, int> resultSet(r2, indices);
        kdtree.index->findNeighbors(resultSet, &point[ 0 ], nanoflann::SearchParameters(0, false));
        if (indices.size() < 3)
            continue;

        std::vector<Eigen::Vector3f> neighbors(indices.size());
        for (int j = 0; j < indices.size(); j++)
            neighbors[ j ] = kdtree.m_data[ indices[ j ].first ];

        // pca
        Eigen::Map<const Eigen::Matrix3Xf> P(neighbors[ 0 ].data(), 3, neighbors.size());
        Eigen::Vector3f                    centroid = P.rowwise().mean();
        Eigen::Matrix3Xf                   centered = P.colwise() - centroid;
        Eigen::Matrix3f                    cov      = centered * centered.transpose();

        // eigvecs sorted in increasing order of eigvals
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eig;
        eig.computeDirect(cov);
        Eigen::Vector3f eval   = eig.eigenvalues();
        int             minInd = 0;
        eval.cwiseAbs().minCoeff(&minInd);
        normal = eig.eigenvectors().col(minInd); // is already normalized
        if (!normal.allFinite())
            continue;
        auto d = -1 * normal.dot(centroid);

        // project query point
        const float     distance = point.dot(normal) + d;
        Eigen::Vector3f mean     = point - distance * normal;

        // Local coordinate system (Darboux frame)
        auto vAxis = normal.unitOrthogonal();
        auto uAxis = normal.cross(vAxis);

        // Perform polynomial fit to update point and normal
        if (order <= 1 && nCoeff > indices.size())
            continue;
        auto weightFunc = [ r2 ](float sqDist) { return std::exp(-sqDist / r2); };

        // Allocate matrices and vectors to hold the data used for the polynomial fit
        Eigen::VectorXf weightVec(neighbors.size());
        Eigen::MatrixXf Ps(nCoeff, neighbors.size());
        Eigen::VectorXf fVec(neighbors.size());
        Eigen::MatrixXf PWeightPt(nCoeff, nCoeff);

        // Update neighborhood, since point was projected, and computing relative
        // positions. Note updating only distances for the weights for speed
        std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>> deMeaned(
            neighbors.size());
        for (int j = 0; j < neighbors.size(); j++) {
            auto &neighbor = neighbors[ j ];
            deMeaned[ j ]  = neighbor - mean;
            weightVec[ j ] = weightFunc(deMeaned[ j ].squaredNorm());
        }

        // Go through neighbors, transform them in the local coordinate system,
        // save height and the evaluation of the polynomial's terms
        for (int j = 0; j < neighbors.size(); j++) {
            // Transforming coordinates
            const float uCoord = deMeaned[ j ].dot(uAxis);
            const float vCoord = deMeaned[ j ].dot(vAxis);
            fVec(j)            = deMeaned[ j ].dot(normal);

            // Compute the polynomial's terms at the current point
            int   idx  = 0;
            float uPow = 1;
            for (int ui = 0; ui <= order; ++ui) {
                float vPow = 1;
                for (int vi = 0; vi <= order - ui; ++vi) {
                    Ps(idx++, j) = uPow * vPow;
                    vPow *= vCoord;
                }
                uPow *= uCoord;
            }
        }

        // Computing coefficients
        Eigen::MatrixXf PWeight = Ps * weightVec.asDiagonal();
        PWeightPt               = PWeight * Ps.transpose();
        Eigen::VectorXf cVec    = PWeight * fVec;
        PWeightPt.llt().solveInPlace(cVec);
        if (!cVec.allFinite())
            continue;

        // project query point
        // Projection onto MLS surface along Darboux normal to the height at (0,0)
        point = mean + (cVec[ 0 ] * normal);
        // Compute tangent vectors using the partial derivates evaluated at (0,0) which is
        // c_vec[order_+1] and c_vec[1]
        normal = (normal - cVec[ order + 1 ] * uAxis - cVec[ 1 ] * vAxis).normalized();

        if (pc.viewPoint.allFinite()) {
            if (check(normal.dot(pc.viewPoint - point)))
                normal = -normal;
        } else {
            if (check(normal.dot(Eigen::Vector3f::UnitZ())))
                normal = -normal;
        }
    }
}

Eigen::Matrix4f transformRT(const Eigen::Vector3f &p, const Eigen::Vector3f &n) {
    float           angle = acos(n.x());    // rotation angle
    Eigen::Vector3f axis(0, n.z(), -n.y()); // rotation axis

    if (n.y() == 0 && n.z() == 0) {
        axis(0) = 0;
        axis(1) = 1;
        axis(2) = 0;
    }

    Eigen::AngleAxisf rotationVector(angle, axis.normalized());
    Eigen::Matrix4f   transform = Eigen::Matrix4f::Identity();
    transform.block(0, 0, 3, 3) = rotationVector.toRotationMatrix();
    transform.block(0, 3, 3, 1) = (-1.f) * transform.block(0, 0, 3, 3) * p;

    return transform;
}

float computeAlpha(Eigen::Matrix4f &rt, const Eigen::Vector3f &p2) {
    Eigen::Vector3f mpt   = rt.topLeftCorner(3, 3) * p2 + rt.topRightCorner(3, 1);
    float           alpha = atan2(-mpt(2), mpt(1));
    if (sin(alpha) * mpt(2) > 0) {
        alpha = -alpha;
    }
    return alpha;
}

std::vector<Pose> sortPoses(std::vector<Pose> poseList) {
    std::sort(poseList.begin(), poseList.end(),
              [](Pose &a, Pose &b) { return a.numVotes > b.numVotes; });

    return poseList;
}

bool comparePose(const Pose &p1, const Pose &p2, float distanceThreshold, float angleThreshold) {
    float d   = (p1.pose.topRightCorner(3, 1) - p2.pose.topRightCorner(3, 1)).norm();
    float phi = std::abs(p1.r.angle() - p2.r.angle());
    return (d < distanceThreshold && phi < angleThreshold);
}

std::vector<std::vector<Pose>> clusterPose(const std::vector<Pose> &poseList,
                                           float distanceThreshold, float angleThreshold) {
    auto sorted = sortPoses(poseList);

    std::vector<std::vector<Pose>> clusters;
    for (auto &pose : sorted) {
        bool assigned = false;
        for (auto &cluster : clusters) {
            auto &poseCenter = cluster[ 0 ];
            if (comparePose(pose, poseCenter, distanceThreshold, angleThreshold)) {
                cluster.push_back(pose);
                assigned = true;
                break;
            }
        }

        if (!assigned)
            clusters.push_back({pose});
    }

    return clusters;
}

std::vector<Pose> clusterPose2(std::vector<Pose> &poseList, Eigen::Vector3f &pos, float threshold) {
    std::vector<Eigen::Vector3f> trans;
    trans.reserve(poseList.size());
    for (auto &pose : poseList) {
        trans.emplace_back(pose.pose.topLeftCorner(3, 3) * pos + pose.pose.topRightCorner(3, 1));
    }

    float             squaredThreshold = threshold * threshold;
    std::vector<bool> used(poseList.size(), false);
    std::vector<Pose> result;
    for (int i = 0; i < poseList.size(); i++) {
        if (used[ i ])
            continue;

        auto poseI = poseList[ i ];

        for (int j = i + 1; j < poseList.size(); j++) {
            if (used[ j ])
                continue;
            if ((trans[ i ] - trans[ j ]).squaredNorm() < squaredThreshold) {
                poseI.numVotes += poseList[ j ].numVotes;
                used[ j ] = true;
            }
        }

        result.push_back(poseI);
    }

    return result;
}

Eigen::Quaternionf avgQuaternionMarkley(const std::vector<Eigen::Quaternionf> &qs) {
    Eigen::Matrix4f A = Eigen::Matrix4f::Zero();
    auto            M = qs.size();
    for (auto &q : qs) {
        Eigen::Vector4f v(q.w(), q.x(), q.y(), q.z());
        A += v * v.transpose();
    }

    A /= (float)M;

    Eigen::EigenSolver<Eigen::Matrix4f> es(A);
    Eigen::MatrixXcf                    evecs =
        es.eigenvectors(); //获取矩阵特征向量4*4，这里定义的MatrixXcd必须有c，表示获得的是complex复数矩阵
    Eigen::MatrixXcf evals = es.eigenvalues(); //获取矩阵特征值 4*1
    Eigen::MatrixXf  evalsReal;                //注意这里定义的MatrixXd里没有c
    evalsReal = evals.real();                  //获取特征值实数部分
    Eigen::MatrixXf::Index evalsMax;
    evalsReal.rowwise().sum().maxCoeff(&evalsMax); //得到最大特征值的位置

    Eigen::Vector4f q;
    q << evecs.real()(0, evalsMax), evecs.real()(1, evalsMax), evecs.real()(2, evalsMax),
        evecs.real()(3, evalsMax); //得到对应特征向量

    return {q[ 0 ], q[ 1 ], q[ 2 ], q[ 3 ]};
}

std::vector<Pose> avgClusters(const std::vector<std::vector<Pose>> &clusters) {
    std::vector<Pose> avg;

    for (auto &cluster : clusters) {
        Eigen::Vector3f p;
        p << 0, 0, 0;
        std::vector<Eigen::Quaternionf> qs;
        float                           votes = 0;
        for (auto &pose : cluster) {
            p += pose.pose.topRightCorner(3, 1);
            votes += pose.numVotes;
            qs.push_back(pose.q);
        }

        Pose pose(votes);
        pose.updatePoseT(p / cluster.size());
        pose.updatePoseQuat(avgQuaternionMarkley(qs));

        avg.push_back(pose);
    }

    return avg;
}

void findClosestPoint(const KDTree &kdtree, const PointCloud &srcPC, std::vector<int> &indices,
                      std::vector<float> &distances) {

    auto               size      = srcPC.size();
    const int          numResult = 1;
    std::vector<int>   indicesTmp(size);
    std::vector<float> distancesTmp(size);

#pragma omp parallel for
    for (int i = 0; i < size; i++) {
        auto                          &point = srcPC.point[ i ];
        std::vector<size_t>            indexes(numResult);
        std::vector<float>             dists(numResult);
        nanoflann::KNNResultSet<float> resultSet(numResult);
        resultSet.init(&indexes[ 0 ], &dists[ 0 ]);
        kdtree.index->findNeighbors(resultSet, &point[ 0 ]);

        indicesTmp[ i ]   = indexes[ 0 ];
        distancesTmp[ i ] = dists[ 0 ];
    }

    indices   = std::move(indicesTmp);
    distances = std::move(distancesTmp);
}

int inliner(const PointCloud &srcPC, const KDTree &kdtree, float inlineDist) {
    std::vector<int>   indices;
    std::vector<float> distances;
    findClosestPoint(kdtree, srcPC, indices, distances);

    int   result            = 0;
    float inlineDistSquared = inlineDist * inlineDist;
    for (auto &dist : distances) {
        if (dist < inlineDistSquared)
            result++;
    }

    return result;
}

uint32_t murmurhash3(const int *key, uint32_t len, uint32_t seed) {
    static const uint32_t c1      = 0xcc9e2d51;
    static const uint32_t c2      = 0x1b873593;
    static const uint32_t r1      = 15;
    static const uint32_t r2      = 13;
    static const uint32_t m       = 5;
    static const uint32_t n       = 0xe6546b64;
    uint32_t              hash    = seed;
    auto                  nBlocks = len / 4;
    auto                 *blocks  = (const uint32_t *)key;

    for (int i = 0; i < nBlocks; i++) {
        uint32_t k = blocks[ i ];
        k *= c1;
        k = (k << r1) | (k >> (32 - r1));
        k *= c2;
        hash ^= k;
        hash = ((hash << r2) | (hash >> (32 - r2))) * m + n;
    }

    auto    *tail = (const uint8_t *)(key + nBlocks * 4);
    uint32_t k1   = 0;

    switch (len & 3) {
        case 3:
            k1 ^= tail[ 2 ] << 16;
        case 2:
            k1 ^= tail[ 1 ] << 8;
        case 1:
            k1 ^= tail[ 0 ];

            k1 *= c1;
            k1 = (k1 << r1) | (k1 >> (32 - r1));
            k1 *= c2;
            hash ^= k1;
    }

    hash ^= len;
    hash ^= (hash >> 16);
    hash *= 0x85ebca6b;
    hash ^= (hash >> 13);
    hash *= 0xc2b2ae35;
    hash ^= (hash >> 16);

    return hash;
}

uint32_t hashPPF(const Eigen::Vector4f &ppfValue, float angleRadians, float distanceStep) {
    const int key[ 4 ] = {int(ppfValue[ 0 ] / angleRadians), int(ppfValue[ 1 ] / angleRadians),
                          int(ppfValue[ 2 ] / angleRadians), int(ppfValue[ 3 ] / distanceStep)};

    return murmurhash3(key, 16, 42);
}

inline float angleA2B(const Eigen::Vector3f &a, const Eigen::Vector3f &b) {
    return atan2((a.cross(b)).norm(), a.dot(b));
}

inline xsimd::batch<float> angleA2B(const xsimd::batch<float> &ax, const xsimd::batch<float> &ay,
                                    const xsimd::batch<float> &az, const xsimd::batch<float> &bx,
                                    const xsimd::batch<float> &by, const xsimd::batch<float> &bz) {
    auto i     = ay * bz - az * by;
    auto j     = az * bx - ax * bz;
    auto k     = ax * by - ay * bx;
    auto cross = xsimd::sqrt(i * i + j * j + k * k);
    auto dot   = ax * bx + ay * by + az * bz;
    return xsimd::atan2(cross, dot);
}

uint32_t computePPF(const Eigen::Vector3f &p1, const Eigen::Vector3f &n1, const Eigen::Vector3f &p2,
                    const Eigen::Vector3f &n2, float angleStep, float distStep) {

    Eigen::Vector3f d  = p2 - p1;
    float           dn = d.norm();
    float           f1, f2, f3;
    if (dn > 0) {
        Eigen::Vector3f dNorm = d / dn;
        f1 = angleA2B(dNorm, n1); // atan2((dNorm.cross(n1)).norm(), dNorm.dot(n1));
        f2 = angleA2B(dNorm, n2); // atan2((dNorm.cross(n2)).norm(), dNorm.dot(n2));
        f3 = angleA2B(n1, n2);    // atan2((n1.cross(n2)).norm(), n1.dot(n2));
    } else {
        f1 = 0;
        f2 = 0;
        f3 = 0;
    }

    return hashPPF({f1, f2, f3, dn}, angleStep, distStep);
}

xsimd::batch<uint32_t> murmurhash3(const std::vector<xsimd::batch<uint32_t>> &data, uint32_t seed) {
    static const auto c1   = xsimd::broadcast<uint32_t>(0xcc9e2d51);
    static const auto c2   = xsimd::broadcast<uint32_t>(0x1b873593);
    static const auto r1   = xsimd::broadcast<uint32_t>(15);
    static const auto r2   = xsimd::broadcast<uint32_t>(13);
    static const auto r3   = xsimd::broadcast<uint32_t>(17);
    static const auto r4   = xsimd::broadcast<uint32_t>(19);
    static const auto r5   = xsimd::broadcast<uint32_t>(16);
    static const auto m    = xsimd::broadcast<uint32_t>(5);
    static const auto n    = xsimd::broadcast<uint32_t>(0xe6546b64);
    static const auto p    = xsimd::broadcast<uint32_t>(0x85ebca6b);
    static const auto q    = xsimd::broadcast<uint32_t>(0xc2b2ae35);
    auto              hash = xsimd::broadcast<uint32_t>(seed);

    static const uint32_t len = 4 * data.size();
    for (auto k : data) {
        k *= c1;
        k = (k << r1) | (k >> r3);
        k *= c2;
        hash ^= k;
        hash = ((hash << r2) | (hash >> r4)) * m + n;
    }

    hash ^= len;
    hash ^= (hash >> r5);
    hash *= p;
    hash ^= (hash >> r2);
    hash *= q;
    hash ^= (hash >> r5);

    return hash;
}

xsimd::batch<uint32_t> hashPPF(const xsimd::batch<float> &f1, const xsimd::batch<float> &f2,
                               const xsimd::batch<float> &f3, const xsimd::batch<float> &dn,
                               float angleStep, float distStep) {
    auto rAngle = xsimd::broadcast<float>(angleStep);
    auto rDist  = xsimd::broadcast<float>(distStep);

    auto dF1 = xsimd::batch_cast<uint32_t>(xsimd::ceil(f1 / rAngle));
    auto dF2 = xsimd::batch_cast<uint32_t>(xsimd::ceil(f2 / rAngle));
    auto dF3 = xsimd::batch_cast<uint32_t>(xsimd::ceil(f3 / rAngle));
    auto dDn = xsimd::batch_cast<uint32_t>(xsimd::ceil(dn / rDist));

    return murmurhash3({dF1, dF2, dF3, dDn}, 42);
}

vectorI computePPF(const Eigen::Vector3f &p1, const Eigen::Vector3f &n1, const vectorF &p2x,
                   const vectorF &p2y, const vectorF &p2z, const vectorF &n2x, const vectorF &n2y,
                   const vectorF &n2z, float angleStep, float distStep) {
    auto                  size      = p2x.size();
    constexpr std::size_t simd_size = xsimd::simd_type<float>::size;
    std::size_t           vec_size  = size - size % simd_size;

    auto rp1x = xsimd::broadcast<float>(p1.x());
    auto rp1y = xsimd::broadcast<float>(p1.y());
    auto rp1z = xsimd::broadcast<float>(p1.z());
    auto rn1x = xsimd::broadcast<float>(n1.x());
    auto rn1y = xsimd::broadcast<float>(n1.y());
    auto rn1z = xsimd::broadcast<float>(n1.z());

    vectorI result(size);
    for (int i = 0; i < vec_size; i += simd_size) {
        auto rp2x = xsimd::load(&p2x[ i ]);
        auto rp2y = xsimd::load(&p2y[ i ]);
        auto rp2z = xsimd::load(&p2z[ i ]);
        auto rn2x = xsimd::load(&n2x[ i ]);
        auto rn2y = xsimd::load(&n2y[ i ]);
        auto rn2z = xsimd::load(&n2z[ i ]);

        auto dx = rp2x - rp1x;
        auto dy = rp2y - rp1y;
        auto dz = rp2z - rp1z;

        auto norm = xsimd::sqrt(dx * dx + dy * dy + dz * dz);
        auto nx   = dx / norm;
        auto ny   = dy / norm;
        auto nz   = dz / norm;

        auto f1 = angleA2B(nx, ny, nz, rn1x, rn1y,
                           rn1z); // xsimd::acos(rn1x * nx + rn1y * ny + rn1z * nz);
        auto f2 = angleA2B(nx, ny, nz, rn2x, rn2y,
                           rn2z); // xsimd::acos(rn2x * nx + rn2y * ny + rn2z * nz);
        auto f3 = angleA2B(rn1x, rn1y, rn1z, rn2x, rn2y,
                           rn2z); // xsimd::acos(rn2x * rn1x + rn2y * rn1y + rn2z * rn1z);

        auto hash = hashPPF(f1, f2, f3, norm, angleStep, distStep);
        xsimd::store(&result[ i ], hash);
    }

    for (int i = vec_size; i < size; i++)
        result[ i ] = computePPF(p1, n1, {p2x[ i ], p2y[ i ], p2z[ i ]},
                                 {n2x[ i ], n2y[ i ], n2z[ i ]}, angleStep, distStep);

    return result;
}

vectorF computeAlpha(Eigen::Matrix4f &rt, const vectorF &p2x, const vectorF &p2y,
                     const vectorF &p2z) {
    // auto r00 = xsimd::broadcast(rt(0, 0));
    // auto r01 = xsimd::broadcast(rt(0, 1));
    // auto r02 = xsimd::broadcast(rt(0, 2));
    auto r10 = xsimd::broadcast(rt(1, 0));
    auto r11 = xsimd::broadcast(rt(1, 1));
    auto r12 = xsimd::broadcast(rt(1, 2));
    auto r20 = xsimd::broadcast(rt(2, 0));
    auto r21 = xsimd::broadcast(rt(2, 1));
    auto r22 = xsimd::broadcast(rt(2, 2));

    // auto t0 = xsimd::broadcast(rt(0, 3));
    auto t1 = xsimd::broadcast(rt(1, 3));
    auto t2 = xsimd::broadcast(rt(2, 3));

    auto inverse = xsimd::broadcast(-1.f);
    // auto zero    = xsimd::broadcast(0.f);

    auto                  size      = p2x.size();
    constexpr std::size_t simd_size = xsimd::simd_type<float>::size;
    std::size_t           vec_size  = size - size % simd_size;
    vectorF               result(size);

    for (int i = 0; i < vec_size; i += simd_size) {
        auto rp2x = xsimd::load(&p2x[ i ]);
        auto rp2y = xsimd::load(&p2y[ i ]);
        auto rp2z = xsimd::load(&p2z[ i ]);

        // auto x = r00 * rp2x + r01 * rp2y + r02 * rp2z + t0;
        auto y = r10 * rp2x + r11 * rp2y + r12 * rp2z + t1;
        auto z = r20 * rp2x + r21 * rp2y + r22 * rp2z + t2;

        auto alpha  = xsimd::atan2(z * inverse, y);
        auto ialpha = alpha * inverse;
        auto t      = xsimd::select(xsimd::sin(alpha) * z > 0, ialpha, alpha);
        xsimd::store(&result[ i ], t);
    }

    for (int i = vec_size; i < size; i++)
        result[ i ] = computeAlpha(rt, {p2x[ i ], p2y[ i ], p2z[ i ]});

    return result;
}

} // namespace ppf