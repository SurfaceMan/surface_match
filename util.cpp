#include <util.h>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <Eigen/StdVector>
#include <fstream>

#include <iostream>

#define _USE_MATH_DEFINES
#include <math.h>

namespace ppf {

PointCloud samplePointCloud(const ppf::PointCloud &pc, float sampleStep, BoxGrid *boxGrid) {
    BoundingBox box;
    if (pc.box.diameter() == 0)
        box = computeBoundingBox(pc);
    else
        box = pc.box;

    float r2      = sampleStep * sampleStep;
    float boxSize = sampleStep / sqrtf(3.0f);
    auto  size    = box.size();
    int   xBins   = ceil(size.x() / boxSize);
    int   yBins   = ceil(size.y() / boxSize);
    int   zBins   = ceil(size.z() / boxSize);

    int maxXIndex = xBins - 1;
    int maxYIndex = yBins - 1;
    int maxZIndex = zBins - 1;

    float xScale = (float)maxXIndex / (float)size.x();
    float yScale = (float)maxYIndex / (float)size.y();
    float zScale = (float)maxZIndex / (float)size.z();

    // std::map<uint64_t, int> map;
    std::vector<std::vector<std::vector<int>>> map;
    {
        std::vector<int>              item1(zBins, BoxGrid::INVALID);
        std::vector<std::vector<int>> item2(yBins, item1);
        map.resize(xBins, item2);
    }
    for (int count = 0; count < pc.point.size(); count++) {
        auto &p = pc.point[ count ];

        Eigen::Vector3i index;
        index.x() = floor(xScale * (p.x() - box.min.x()));
        index.y() = floor(yScale * (p.y() - box.min.y()));
        index.z() = floor(zScale * (p.z() - box.min.z()));

        // find neighbor
        bool sampleModel = true;
        int  iBegin      = std::max(0, index.x() - 2);
        int  iEnd        = std::min(index.x() + 2, xBins);
        int  jBegin      = std::max(0, index.y() - 2);
        int  jEnd        = std::min(index.y() + 2, yBins);
        int  kBegin      = std::max(0, index.z() - 2);
        int  kEnd        = std::min(index.z() + 2, zBins);
        for (int i = iBegin; i < iEnd; i++) {
            for (int j = jBegin; j < jEnd; j++) {
                for (int k = kBegin; k < kEnd; k++) {
                    int pointIndex = map[ i ][ j ][ k ];
                    if (pointIndex == BoxGrid::INVALID)
                        continue;

                    float dist2 = (p - pc.point[ pointIndex ]).squaredNorm();
                    if (dist2 < r2) {
                        sampleModel = false;
                        break;
                    }
                }
            }
        }

        if (sampleModel)
            map[ index.x() ][ index.y() ][ index.z() ] = count;
    }

    PointCloud                                 result;
    std::vector<Eigen::Vector3i>               grid;
    std::vector<std::vector<std::vector<int>>> idx;
    {
        std::vector<int>              item1(zBins, BoxGrid::INVALID);
        std::vector<std::vector<int>> item2(yBins, item1);
        idx.resize(xBins, item2);
    }
    bool hasNormal  = !pc.normal.empty();
    int  pointCount = 0;
    for (int k = 0; k < zBins; k++) {
        for (int j = 0; j < yBins; j++) {
            for (int i = 0; i < xBins; i++) {
                int count = map[ i ][ j ][ k ];
                if (count == BoxGrid::INVALID)
                    continue;

                grid.emplace_back(i, j, k);
                idx[ i ][ j ][ k ] = pointCount++;

                result.point.push_back(pc.point[ count ]);
                if (hasNormal)
                    result.normal.push_back(pc.normal[ count ].normalized());
            }
        }
    }
    result.box = box;

    if (boxGrid) {
        boxGrid->grid  = std::move(grid);
        boxGrid->index = std::move(idx);
        boxGrid->step  = boxSize;
        boxGrid->xBins = xBins;
        boxGrid->yBins = yBins;
        boxGrid->zBins = zBins;
    }

    return result;
}

PointCloud samplePointCloud2(const ppf::PointCloud &pc, float sampleStep, KDTree *tree) {
    KDTree *kdtree     = tree;
    bool    needDelete = false;

    if (!kdtree) {
        kdtree     = new KDTree(3, pc.point);
        needDelete = true;
    }

    auto                                size = pc.point.size();
    std::vector<bool>                   keep(size, true);
    const std::vector<Eigen::Vector3f> &points = pc.point;
    auto                                radius = sampleStep * sampleStep;

#pragma parallel for
    for (std::size_t i = 0; i < size; i++) {
        if (!keep[ i ])
            continue;

        auto                                      &point = pc.point[ i ];
        std::vector<std::pair<std::size_t, float>> indices;
        kdtree->index->radiusSearch(&point[ 0 ], radius, indices, nanoflann::SearchParams());

        for (std::size_t j = 1; j < indices.size(); j++)
            keep[ indices[ j ].first ] = false;
    }

    PointCloud result;
    bool       hasNormal = pc.hasNormal();
    for (std::size_t i = 0; i < size; i++) {
        if (!keep[ i ])
            continue;

        result.point.emplace_back(pc.point[ i ]);
        if (hasNormal)
            result.normal.emplace_back(pc.normal[ i ].normalized());
    }

    result.box = pc.box;
    if (result.box.diameter() == 0)
        result.box = computeBoundingBox(result);

    return result;
}

BoundingBox computeBoundingBox(const ppf::PointCloud &pc) {
    Eigen::Vector3f min = pc.point[ 0 ];
    Eigen::Vector3f max = min;

    // bounding box
    for (auto &p : pc.point) {
        if (p.x() > max.x())
            max.x() = p.x();
        else if (p.x() < min.x())
            min.x() = p.x();
        if (p.y() > max.y())
            max.y() = p.y();
        else if (p.y() < min.y())
            min.y() = p.y();
        if (p.z() > max.z())
            max.z() = p.z();
        else if (p.z() < min.z())
            min.z() = p.z();
    }

    return {min, max};
}

PointCloud transformPointCloud(const ppf::PointCloud &pc, const Eigen::Matrix4f &pose) {
    auto size      = pc.point.size();
    auto hasNormal = !pc.normal.empty();

    PointCloud result;
    result.point.reserve(size);
    if (hasNormal)
        result.normal.reserve(size);

    Eigen::Matrix3f rMat;
    rMat << pose(0, 0), pose(0, 1), pose(0, 2), pose(1, 0), pose(1, 1), pose(1, 2), pose(2, 0),
        pose(2, 1), pose(2, 2);

    for (auto &point : pc.point) {
        Eigen::Vector4f p(point.x(), point.y(), point.z(), 1);
        p = pose * p;

        result.point.emplace_back(p[ 0 ], p[ 1 ], p[ 2 ]);
    }
    if (hasNormal) {
        for (auto &n : pc.normal) {
            result.normal.push_back((rMat * n).normalized());
        }
    }

    result.box = computeBoundingBox(result);
    return result;
}

std::vector<Eigen::Vector3f> estimateNormal(const ppf::PointCloud &pc) {
    return {};
}

std::vector<Eigen::Vector3f> estimateNormal(const ppf::PointCloud &pc, const ppf::PointCloud &ref) {
    return {};
}

Eigen::Vector4f computePPF(const Eigen::Vector3f &p1, const Eigen::Vector3f &p2,
                           const Eigen::Vector3f &n1, const Eigen::Vector3f &n2) {

    Eigen::Vector3f d  = p2 - p1;
    float           dn = d.norm();
    float           f1, f2, f3;
    if (dn > 0) {
        Eigen::Vector3f dNorm = d / dn;
        f1                    = atan2((dNorm.cross(n1)).norm(), dNorm.dot(n1));
        f2                    = atan2((dNorm.cross(n2)).norm(), dNorm.dot(n2));
        f3                    = atan2((n1.cross(n2)).norm(), n1.dot(n2));
    } else {
        f1 = 0;
        f2 = 0;
        f3 = 0;
    }

    return {f1, f2, f3, dn};
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

void transformRT(const Eigen::Vector3f &p, const Eigen::Vector3f &n, Eigen::Matrix3f &R,
                 Eigen::Vector3f &t) {
    float           angle = acos(n.x());    // rotation angle
    Eigen::Vector3f axis(0, n.z(), -n.y()); // rotation axis

    if (n.y() == 0 && n.z() == 0) {
        axis(0) = 0;
        axis(1) = 1;
        axis(2) = 0;
    }

    Eigen::AngleAxisf rotationVector(angle, axis.normalized());
    R = rotationVector.toRotationMatrix(); // rotation matrix
    t = (-1) * R * p;
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

float computeAlpha(const Eigen::Vector3f &p1, const Eigen::Vector3f &p2,
                   const Eigen::Vector3f &n1) {
    Eigen::Matrix3f R;
    Eigen::Vector3f t;

    transformRT(p1, n1, R, t);
    Eigen::Vector3f mpt   = R * p2 + t;
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
    float d   = (p1.pose.translation() - p2.pose.translation()).norm();
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
    Eigen::Vector4f p(pos.x(), pos.y(), pos.z(), 1);

    std::vector<Eigen::Vector4f> trans;
    trans.reserve(poseList.size());
    for (auto &pose : poseList) {
        trans.emplace_back(pose.pose.matrix() * p);
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
            p += pose.pose.translation();
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

PointCloud loadText(const std::string &filename) {
    PointCloud      result;
    Eigen::Vector3f p;
    Eigen::Vector3f n;

    FILE *file = fopen(filename.c_str(), "r");
    while (!feof(file)) {
        fscanf(file, "%f %f %f %f %f %f", &p.x(), &p.y(), &p.z(), &n.x(), &n.y(), &n.z());
        result.point.push_back(p);
        result.normal.push_back(n);
    }
    fclose(file);

    result.box = computeBoundingBox(result);
    return result;
}

void saveText(const std::string &filename, const PointCloud &pc) {
    std::ofstream out(filename);
    for (int i = 0; i < pc.point.size(); i++) {
        auto &p = pc.point[ i ];
        auto &n = pc.normal[ i ];
        out << p.x() << '\t' << p.y() << '\t' << p.z() << '\t' << n.x() << '\t' << n.y() << '\t'
            << n.z() << '\n';
    }
    out.close();
}

void findClosestPoint(const KDTree &kdtree, const PointCloud &srcPC, std::vector<int> &indices,
                      std::vector<float> &distances) {

    const int          numResult = 1;
    std::vector<int>   indicesTmp;
    std::vector<float> distancesTmp;
    for (auto &point : srcPC.point) {
        std::vector<size_t>            indexes(numResult);
        std::vector<float>             dists(numResult);
        nanoflann::KNNResultSet<float> resultSet(1);
        resultSet.init(&indexes[ 0 ], &dists[ 0 ]);

        kdtree.index->findNeighbors(resultSet, &point[ 0 ], nanoflann::SearchParams());

        indicesTmp.push_back(indexes[ 0 ]);
        distancesTmp.push_back(dists[ 0 ]);
    }

    indices   = std::move(indicesTmp);
    distances = std::move(distancesTmp);
}

void getCoordinateSystemOnPlane(const Eigen::Vector3f &query, Eigen::Vector4f &u,
                                Eigen::Vector4f &v) {
    const Eigen::Vector4f vector = Eigen::Vector4f(query.x(), query.y(), query.z(), 0);
    v                            = vector.unitOrthogonal();
    u                            = vector.cross3(v);
}

bool isBoundary(const PointCloud &srcPC, const Eigen::Vector3f &point,
                const Eigen::Vector3f                            &normal,
                const std::vector<std::pair<std::size_t, float>> &indices, float angleThreshold) {
    Eigen::Vector4f u = Eigen::Vector4f::Zero();
    Eigen::Vector4f v = Eigen::Vector4f::Zero();
    getCoordinateSystemOnPlane(normal, u, v);
    std::vector<float> angles;

    for (auto &idx : indices) {
        auto &p     = srcPC.point[ idx.first ];
        auto  delta = p - point;
        if (delta == Eigen::Vector3f::Zero())
            continue;

        const Eigen::Vector4f vec = Eigen::Vector4f(delta.x(), delta.y(), delta.z(), 0);
        angles.push_back(atan2(v.dot(vec), u.dot(vec)));
    }

    if (angles.empty())
        return false;
    std::sort(angles.begin(), angles.end());
    // Compute the maximal angle difference between two consecutive angles
    float dif;
    float max_dif = 0;
    for (size_t i = 0; i < angles.size() - 1; ++i) {
        dif = abs(angles[ i + 1 ] - angles[ i ]);
        if (max_dif < dif) {
            max_dif = dif;
        }
    }

    // Get the angle difference between the last and the first
    dif = abs(2 * M_PI - angles[ angles.size() - 1 ] + angles[ 0 ]);
    if (max_dif < dif)
        max_dif = dif;

    return max_dif > angleThreshold;
}

std::vector<std::size_t> findEdge(const KDTree &kdtree, const PointCloud &srcPC, float radius,
                                  float angleThreshold) {
    float                    radius2 = radius * radius;
    std::vector<std::size_t> results;

#pragma omp parallel for
    for (int i = 0; i < srcPC.point.size(); i++) {
        auto &p = srcPC.point[ i ];
        auto &n = srcPC.normal[ i ];

        std::vector<std::pair<std::size_t, float>> indices;
        auto                                       searched =
            kdtree.index->radiusSearch(&p[ 0 ], radius2, indices, nanoflann::SearchParams());
        if (searched < 3)
            continue;

        if (isBoundary(srcPC, p, n, indices, angleThreshold)) {
#pragma omp critical
            { results.push_back(i); }
        }
    }

    return results;
}

} // namespace ppf