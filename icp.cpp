#include <icp.h>

#include <KDTreeVectorOfVectorsAdaptor.h>
#include <util.h>

namespace ppf {
struct ICP::IMPL {
public:
    int   maxIterations;
    float tolerance;
    float rejectionScale;

    IMPL(int iterations_, float tolerance_, float rejectionScale_)
        : maxIterations(iterations_)
        , tolerance(tolerance_)
        , rejectionScale(rejectionScale_) {
    }
};

ICP::ICP(const int iterations, const float tolerance, const float rejectionScale)
    : impl_(new IMPL(iterations, tolerance, rejectionScale)) {
}

ICP::~ICP() {
}

typedef std::vector<Eigen::Vector3f>                   vectors_t;
typedef KDTreeVectorOfVectorsAdaptor<vectors_t, float> kd_tree_t;

void findClosestPoint(const kd_tree_t &kdtree, const PointCloud &srcPC, std::vector<int> &indicies,
                      std::vector<float> &distances) {

    const int          numResult = 1;
    std::vector<int>   indiciesTmp;
    std::vector<float> distancesTmp;
    for (auto &point : srcPC.point) {
        std::vector<size_t>            indexes(numResult);
        std::vector<float>             dists(numResult);
        nanoflann::KNNResultSet<float> resultSet(1);
        resultSet.init(&indexes[ 0 ], &dists[ 0 ]);

        kdtree.index->findNeighbors(resultSet, &point[ 0 ], nanoflann::SearchParams());

        indiciesTmp.push_back(indexes[ 0 ]);
        distancesTmp.push_back(dists[ 0 ]);
    }

    indicies  = std::move(indiciesTmp);
    distances = std::move(distancesTmp);
}

Eigen::Matrix3f xyz2Matrix(float rx, float ry, float rz) {
    Eigen::AngleAxisf AngleRx(rx, Eigen::Vector3f::UnitX());
    Eigen::AngleAxisf AngleRy(ry, Eigen::Vector3f::UnitY());
    Eigen::AngleAxisf AngleRz(rz, Eigen::Vector3f::UnitZ());

    Eigen::Matrix3f org;
    org = AngleRx * AngleRy * AngleRz;
    return org;
}

Eigen::Matrix4f minimizePointToPlaneMetric(const PointCloud &srcPC, const PointCloud &dstPC) {
    int size = srcPC.point.size();

    Eigen::MatrixXf A = Eigen::MatrixXf::Zero(size, 6);
    Eigen::MatrixXf B = Eigen::MatrixXf::Zero(size, 1);

    for (int i = 0; i < size; i++) {
        auto &p1 = srcPC.point[ i ];
        auto &p2 = dstPC.point[ i ];
        auto &n2 = dstPC.normal[ i ];

        auto sub  = p2 - p1;
        auto axis = p1.cross(n2);
        auto v    = sub.dot(n2);

        A.block<1, 3>(i, 0) = axis;
        A.block<1, 3>(i, 3) = n2;
        B(i, 0)             = v;
    }

    Eigen::JacobiSVD<Eigen::MatrixXf> svd(A, Eigen::ComputeThinU | Eigen::ComputeThinV);
    auto                              x = svd.solve(B);

    Eigen::Vector3f rpy = x.block<3, 1>(0, 0);
    Eigen::Vector3f t   = x.block<3, 1>(3, 0);

    Eigen::Matrix4f result   = Eigen::Matrix4f::Identity();
    result.block<3, 3>(0, 0) = xyz2Matrix(rpy.x(), rpy.y(), rpy.z());
    result.block<3, 1>(0, 3) = t.transpose();

    return result;
}

float meanDiff(const PointCloud &srcPC, const PointCloud &dstPC) {
    float diff = 0;

    int size = srcPC.point.size();
    for (int i = 0; i < size; i++) {
        auto &p1 = srcPC.point[ i ];
        auto &n1 = srcPC.normal[ i ];
        auto &p2 = dstPC.point[ i ];
        auto &n2 = dstPC.normal[ i ];

        diff += sqrtf((n2 - n1).squaredNorm() + (p2 - p1).squaredNorm()) / size;
    }

    return diff;
}

float mid(const std::vector<float> &distances) {
    auto tmp = distances;
    std::sort(tmp.begin(), tmp.end());
    auto index = tmp.size() / 2;
    return tmp[ index ];
}

float getRejectThreshold(const std::vector<float> &distances, float rejectScale) {
    auto               midVal = mid(distances);
    auto               size   = distances.size();
    std::vector<float> tmp(size);
    for (size_t i = 0; i < size; i++) {
        tmp[ i ] = distances[ i ] - midVal;
    }

    auto  s         = 1.48257968f * mid(tmp);
    float threshold = rejectScale * s + midVal;

    return threshold;
}

int ICP::registerModelToScene(const PointCloud &srcPC, const PointCloud &dstPC, float &residual,
                              Eigen::Matrix4f &pose) {
    int        n               = srcPC.point.size();
    const bool useRobustReject = impl_->rejectionScale > 0;

    auto  srcTmp = srcPC;
    auto &dstTmp = dstPC;

    // initialize pose
    pose            = Eigen::Matrix4f::Identity();
    float tolerance = impl_->tolerance;
    int   maxIter   = impl_->maxIterations;

    float TolP = tolerance;

    float     valOld = 9999999999;
    float     valMin = 9999999999;
    int       iter   = 0;
    kd_tree_t kdtree(3, dstTmp.point, 10);
    kdtree.index->buildIndex();

    while (iter++ < maxIter) {
        std::vector<int>   indicies;
        std::vector<float> distances;

        findClosestPoint(kdtree, srcTmp, indicies, distances);

        std::map<int, std::vector<int>> map;
        for (int i = 0; i < indicies.size(); i++) {
            auto index = indicies[ i ];
            map[ index ].push_back(i);
        }

        // limit distance
        if (useRobustReject) {
            map.clear();
            auto threshold = getRejectThreshold(distances, impl_->rejectionScale);
            for (size_t i = 0; i < distances.size(); i++) {
                auto &distance = distances[ i ];
                if (distance > threshold)
                    continue;

                auto index = indicies[ i ];
                map[ index ].push_back(i);
            }
        }

        // find model-scene closest point pair
        std::vector<std::pair<int, int>> modelScenePair; //[model_index, scene_index];
        for (auto &node : map) {
            int sceneIndex = node.first;
            int modelIndex = node.second[ 0 ];
            int minDist    = distances[ modelIndex ];

            for (int i = 1; i < node.second.size(); i++) {
                auto &index    = node.second[ i ];
                auto &distance = distances[ index ];
                if (distance < minDist) {
                    minDist    = distance;
                    modelIndex = index;
                }
            }

            modelScenePair.emplace_back(modelIndex, sceneIndex);
        }

        if (modelScenePair.size() < 6) {
            // too few pairs!!!
        }

        PointCloud src;
        PointCloud dst;
        for (auto &item : modelScenePair) {
            src.point.push_back(srcTmp.point[ item.first ]);
            src.normal.push_back(srcTmp.normal[ item.first ]);
            dst.point.push_back(dstTmp.point[ item.second ]);
            dst.normal.push_back(dstTmp.normal[ item.second ]);
        }

        auto p    = minimizePointToPlaneMetric(src, dst);
        auto pct  = transformPointCloud(src, pose);
        auto fval = meanDiff(pct, dst);
        auto perc = fval / valOld;
        valOld    = fval;
        if (fval < valMin)
            valMin = fval;

        residual = valMin;
        pose     = p * pose;

        if (perc < (1 + TolP) || perc > (1 - TolP))
            break;

        srcTmp = transformPointCloud(srcTmp, pose);
    }

    return 0;
}

int ICP::registerModelToScene(const PointCloud &srcPC, const PointCloud &dstPC,
                              std::vector<float> &residual, std::vector<Eigen::Matrix4f> &pose) {
    return 0;
}

} // namespace ppf