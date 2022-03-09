#include <icp.h>

#include <KDTreeVectorOfVectorsAdaptor.h>
#include <util.h>

namespace ppf {
ConvergenceCriteria::ConvergenceCriteria(int iterations_, float rejectDist_, float inlinerDist_,
                                         float mseMin_, float mseMax_, float tolerance_)
    : iterations(iterations_)
    , rejectDist(rejectDist_)
    , inlinerDist(inlinerDist_)
    , mseMin(mseMin_)
    , mseMax(mseMax_)
    , tolerance(tolerance_) {
}

ConvergenceResult::ConvergenceResult()
    : converged(false)
    , type(ConvergenceType::ITER)
    , mse(std::numeric_limits<float>::max())
    , convergeRate(1)
    , iterations(0)
    , pose(Eigen::Matrix4f::Identity())
    , inliner(0) {
}

ICP::ICP(ConvergenceCriteria criteria)
    : criteria_(criteria) {
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
    auto size = srcPC.point.size();

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
        tmp[ i ] = abs(distances[ i ] - midVal);
    }

    auto  s         = 1.48257968f * mid(tmp);
    float threshold = rejectScale * s + midVal;

    return threshold;
}

std::vector<std::pair<int, int>> findCorresponds(const PointCloud &srcPC, const PointCloud &dstPC,
                                                 const KDTree &kdtree, float rejectDist) {
    std::vector<int>   indicies;
    std::vector<float> distances;

    findClosestPoint(kdtree, srcPC, indicies, distances);

    std::map<int, std::vector<int>> map;
    // limit distance
    for (size_t i = 0; i < distances.size(); i++) {
        auto &distance = distances[ i ];
        if (distance > rejectDist)
            continue;

        auto index = indicies[ i ];
        map[ index ].push_back(i);
    }

    // find the closest model-scene point pair
    std::vector<std::pair<int, int>> modelScenePair; //[model_index, scene_index];
    for (auto &node : map) {
        int sceneIndex = node.first;
        int modelIndex =
            *std::min_element(node.second.begin(), node.second.end(),
                              [ & ](int &a, int &b) { return distances[ a ] < distances[ b ]; });

        modelScenePair.emplace_back(modelIndex, sceneIndex);
    }

    return modelScenePair;
}

struct IterResult {
    std::size_t     validPairs = 0;
    Eigen::Matrix4f pose       = Eigen::Matrix4f::Identity();
    float           mse        = std::numeric_limits<float>::max();
};

IterResult iteration(const PointCloud &srcPC, const PointCloud &dstPC, const KDTree &kdtree,
                     float rejectDist) {
    auto modelScenePair = findCorresponds(srcPC, dstPC, kdtree, rejectDist);
    if (modelScenePair.size() < 6)
        return IterResult{modelScenePair.size()};

    PointCloud src;
    PointCloud dst;
    for (auto &item : modelScenePair) {
        src.point.push_back(srcPC.point[ item.first ]);
        src.normal.push_back(srcPC.normal[ item.first ]);
        dst.point.push_back(dstPC.point[ item.second ]);
        dst.normal.push_back(dstPC.normal[ item.second ]);
    }

    auto p   = minimizePointToPlaneMetric(src, dst);
    auto pct = transformPointCloud(src, p);

    std::vector<int>   indices2;
    std::vector<float> distances2;
    PointCloud         dst2;
    findClosestPoint(kdtree, pct, indices2, distances2);
    for (auto &idx : indices2) {
        dst2.point.push_back(dstPC.point[ idx ]);
        dst2.normal.push_back(dstPC.normal[ idx ]);
    }

    float mse = 0;
    for (auto &dist : distances2)
        mse += sqrtf(dist);
    mse /= (float)distances2.size();

    return IterResult{modelScenePair.size(), p, mse};
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

ConvergenceResult ICP::regist(const PointCloud &src, const PointCloud &dst,
                              const Eigen::Matrix4f &initPose) const {
    return regist(src, dst, std::vector<Eigen::Matrix4f>{initPose})[ 0 ];
}

ConvergenceResult ICP::regist(const PointCloud &src, const PointCloud &dst, const KDTree &kdtree,
                              const Eigen::Matrix4f &initPose) const {
    return regist(src, dst, kdtree, std::vector<Eigen::Matrix4f>{initPose})[ 0 ];
}

std::vector<ConvergenceResult> ICP::regist(const PointCloud &src, const PointCloud &dst,
                                           const std::vector<Eigen::Matrix4f> &initPoses) const {
    // initialize
    KDTree kdtree(dst.point, 10);
    return regist(src, dst, kdtree, initPoses);
}

std::vector<ConvergenceResult> ICP::regist(const PointCloud &src, const PointCloud &dst,
                                           const KDTree                       &kdtree,
                                           const std::vector<Eigen::Matrix4f> &initPoses) const {
    if (!src.hasNormal() || !dst.hasNormal())
        throw std::runtime_error("PointCloud empty or no normal at ICP::regist");

    if (criteria_.iterations < 1 || criteria_.inlinerDist < 0 || criteria_.mseMin < 0 ||
        criteria_.mseMax < criteria_.mseMin || criteria_.tolerance > 1 || criteria_.tolerance < 0 ||
        criteria_.rejectDist < 0)
        throw std::runtime_error("Invalid ConvergenceCriteria at ICP::regist");

    std::vector<ConvergenceResult> results(initPoses.size());

    for (int i = 0; i < initPoses.size(); i++) {
        auto &initPose = initPoses[ i ];

        ConvergenceResult result;
        result.pose = initPose;
        auto srcTmp = initPose.isIdentity() ? src : transformPointCloud(src, initPose);

        while (result.iterations < criteria_.iterations) {
            auto tmpResult = iteration(srcTmp, dst, kdtree, criteria_.rejectDist);

            bool converged = (tmpResult.validPairs > 6) && (tmpResult.mse < criteria_.mseMax);
            if (converged)
                result.converged = true;

            bool stop = false;
            if (tmpResult.validPairs < 6) {
                result.type = ConvergenceType::NO_CORRESPONDS;
                stop        = true;
            } else {
                if (tmpResult.mse < criteria_.mseMin) {
                    result.type = ConvergenceType::MSE;
                    stop        = true;
                }

                float convergeRate = (result.mse - tmpResult.mse) / result.mse;
                if (convergeRate < criteria_.tolerance) {
                    result.type = ConvergenceType::CONVERGE_RATE;
                    stop        = true;
                }

                if (result.iterations++ >= criteria_.iterations) {
                    result.type = ConvergenceType::ITER;
                    stop        = true;
                }

                if (convergeRate > 0) {
                    result.mse          = tmpResult.mse;
                    result.convergeRate = convergeRate;
                    result.pose         = tmpResult.pose * result.pose;
                }
            }

            if (stop) {
                auto pct       = transformPointCloud(src, result.pose);
                result.inliner = inliner(pct, kdtree, criteria_.inlinerDist);
                break;
            }

            srcTmp = transformPointCloud(srcTmp, tmpResult.pose);
        }

        results[ i ] = result;
    }

    return results;
}

} // namespace ppf