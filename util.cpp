#include <util.h>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <Eigen/StdVector>
#include <fstream>

#include <iostream>

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

    float xScale = maxXIndex / size.x();
    float yScale = maxYIndex / size.y();
    float zScale = maxZIndex / size.z();

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

    return BoundingBox(min, max);
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

    return Eigen::Vector4f(f1, f2, f3, dn);
}

uint32_t murmurhash3(const int *key, uint32_t len, uint32_t seed) {
    static const uint32_t c1      = 0xcc9e2d51;
    static const uint32_t c2      = 0x1b873593;
    static const uint32_t r1      = 15;
    static const uint32_t r2      = 13;
    static const uint32_t m       = 5;
    static const uint32_t n       = 0xe6546b64;
    uint32_t              hash    = seed;
    const int             nBlocks = len / 4;
    const uint32_t       *blocks  = (const uint32_t *)key;

    for (int i = 0; i < nBlocks; i++) {
        uint32_t k = blocks[ i ];
        k *= c1;
        k = (k << r1) | (k >> (32 - r1));
        k *= c2;
        hash ^= k;
        hash = ((hash << r2) | (hash >> (32 - r2))) * m + n;
    }

    const uint8_t *tail = (const uint8_t *)(key + nBlocks * 4);
    uint32_t       k1   = 0;

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
    } else {
        float axisNorm = axis.norm();
        if (axisNorm > 0)
            axis /= axisNorm;
    }

    Eigen::AngleAxisf rotationVector(angle, axis.normalized());
    R = Eigen::Matrix3f::Identity();
    R = rotationVector.toRotationMatrix(); // rotation matrix
    t = (-1) * R * p;
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
        trans.push_back(pose.pose.matrix() * p);
    }

    float             squaredThre = threshold * threshold;
    std::vector<bool> used(poseList.size(), false);
    std::vector<Pose> result;
    for (int i = 0; i < poseList.size(); i++) {
        if (used[ i ])
            continue;

        auto poseI = poseList[ i ];

        for (int j = i + 1; j < poseList.size(); j++) {
            if (used[ j ])
                continue;
            if ((trans[ i ] - trans[ j ]).squaredNorm() < squaredThre) {
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
    int             M = qs.size();
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

    return Eigen::Quaternionf(q[ 0 ], q[ 1 ], q[ 2 ], q[ 3 ]);
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

Eigen::Vector3f mean(const PointCloud &pc) {
    Eigen::Vector3f point;
    for (auto &p : pc.point) {
        point += p;
    }

    return point / pc.point.size();
}

void subtract(PointCloud &pc, const Eigen::Vector3f &mean) {
    for (auto &p : pc.point) {
        p -= mean;
    }
}

float distToOrigin(PointCloud &pc) {
    int   size  = pc.point.size();
    float scale = 1.0f / (float)size;
    float dist  = 0;
    for (auto &p : pc.point) {
        dist += p.norm() * scale;
    }

    return dist;
}

void scale(PointCloud &pc, float scale_) {
    for (auto &p : pc.point) {
        p *= scale_;
    }
}

} // namespace ppf