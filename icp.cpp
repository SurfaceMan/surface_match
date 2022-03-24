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

#pragma omp parallel for
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

std::pair<PointCloud, PointCloud> findCorresponds(const PointCloud &srcPC, const PointCloud &dstPC,
                                                  const KDTree &kdtree, float rejectDist) {
    std::vector<int>   indicies;
    std::vector<float> distances;
    findClosestPoint(kdtree, srcPC, indicies, distances);

    // limit distance
    std::vector<std::pair<int, std::vector<int>>> map;
    for (int i = 0; i < distances.size(); i++) {
        auto &distance = distances[ i ];
        if (distance > rejectDist)
            continue;

        auto index = indicies[ i ];
        bool found = false;
        for (auto &node : map) {
            if (node.first != index)
                continue;
            node.second.push_back(i);
            found = true;
            break;
        }

        if (found)
            continue;
        map.push_back({index, {i}});
    }

    // find the closest model-scene point pair
    auto       size = map.size();
    PointCloud src;
    src.point.resize(size);
    src.normal.resize(size);
    PointCloud dst;
    dst.point.resize(size);
    dst.normal.resize(size);
#pragma omp parallel for
    for (int i = 0; i < size; i++) {
        auto &node       = map[ i ];
        int   sceneIndex = node.first;
        int   modelIndex = *std::min_element(node.second.begin(), node.second.end());

        src.point[ i ]  = srcPC.point[ modelIndex ];
        src.normal[ i ] = srcPC.normal[ modelIndex ];
        dst.point[ i ]  = dstPC.point[ sceneIndex ];
        dst.normal[ i ] = dstPC.normal[ sceneIndex ];
    }

    return {std::move(src), std::move(dst)};
}

struct IterResult {
    std::size_t     validPairs = 0;
    Eigen::Matrix4f pose       = Eigen::Matrix4f::Identity();
    float           mse        = std::numeric_limits<float>::max();
};

IterResult iteration(const PointCloud &srcPC, const PointCloud &dstPC, const KDTree &kdtree,
                     float rejectDist) {
    auto modelScenePair = findCorresponds(srcPC, dstPC, kdtree, rejectDist);
    auto size           = modelScenePair.first.size();
    if (size < 6)
        return IterResult{size};

    auto &src = modelScenePair.first;
    auto &dst = modelScenePair.second;
    auto  p   = minimizePointToPlaneMetric(src, dst);
    auto  pct = transformPointCloud(src, p);

    std::vector<int>   indices2;
    std::vector<float> distances2;
    findClosestPoint(kdtree, pct, indices2, distances2);

    float mse = 0;
    for (auto &dist : distances2)
        mse += sqrtf(dist);
    mse /= (float)distances2.size();

    return IterResult{modelScenePair.first.size(), p, mse};
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