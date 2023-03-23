#include <icp.h>

#include <privateUtil.h>
#include <util.h>

namespace ppf {
ConvergenceCriteria::ConvergenceCriteria(int iterations_, float rejectDist_, float mseMin_,
                                         float mseMax_, float tolerance_)
    : iterations(iterations_)
    , rejectDist(rejectDist_)
    , mseMin(mseMin_)
    , mseMax(mseMax_)
    , tolerance(tolerance_) {
}

ConvergenceResult::ConvergenceResult()
    : pose(Eigen::Matrix4f::Identity())
    , type(ConvergenceType::ITER)
    , mse(std::numeric_limits<float>::max())
    , convergeRate(1)
    , iterations(0)
    , converged(false) {
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

void symmetric_rigid_matching(const Eigen::MatrixXf &P, const Eigen::MatrixXf &Q,
                              const Eigen::MatrixXf &NP, const Eigen::MatrixXf &NQ,
                              Eigen::Matrix3f &R, Eigen::RowVector3f &t) {

    // normalize point sets
    Eigen::RowVector3f Pmean = P.colwise().mean();
    Eigen::RowVector3f Qmean = Q.colwise().mean();
    Eigen::MatrixXf    Pbar  = P.rowwise() - Pmean;
    Eigen::MatrixXf    Qbar  = Q.rowwise() - Qmean;

    // sum of normals
    Eigen::MatrixXf N = NP + NQ;

    // compute A and b of linear system
    int             num_points = P.rows();
    Eigen::MatrixXf A          = Eigen::MatrixXf::Zero(6, 6);
    Eigen::VectorXf b          = Eigen::VectorXf::Zero(6, 1);
    for (int i = 0; i < num_points; ++i) {
        Eigen::MatrixXf x_i = Eigen::MatrixXf(6, 1);
        Eigen::Vector3f n_i = N.row(i);
        Eigen::Vector3f p_i = Pbar.row(i);
        Eigen::Vector3f q_i = Qbar.row(i);
        double          b_i = (p_i - q_i).dot(n_i);
        x_i << (p_i + q_i).cross(n_i), n_i;
        A += x_i * x_i.transpose();
        b += b_i * x_i;
    }

    // solve linear equation
    Eigen::VectorXf u = A.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(-b);

    Eigen::Vector3f a_tilda = u.head(3); // scaled rotation axis
    Eigen::Vector3f t_tilda = u.tail(3); // scaled translation

    // compute intermediate rotation
    double          theta = atan(a_tilda.norm()); // rotation angle
    Eigen::Vector3f a     = a_tilda.normalized(); // normalized rotation axis
    Eigen::Matrix3f W;
    W << 0, -a(2), a(1), a(2), 0, -a(0), -a(1), a(0), 0;
    Eigen::Matrix3f intermediate_R =
        Eigen::Matrix3f::Identity() + sin(theta) * W + (1 - cos(theta)) * (W * W);

    // compose translations and rotations
    Eigen::Vector3f t1 = -Pmean.transpose();
    Eigen::Vector3f t2 = cos(theta) * t_tilda;
    Eigen::Vector3f t3 = Qmean.transpose();
    R                  = intermediate_R * intermediate_R;
    t                  = (intermediate_R * intermediate_R * t1) + (intermediate_R * t2) + t3;
}

Eigen::Matrix4f minimizePointToPlaneMetric(
    const PointCloud &srcPC, const PointCloud &dstPC,
    const std::pair<std::vector<std::size_t>, std::vector<std::size_t>> &modelScenePair) {
    auto size = modelScenePair.first.size();

    auto &modelIdx = modelScenePair.first;
    auto &sceneIdx = modelScenePair.second;

    Eigen::MatrixXf P  = Eigen::MatrixXf::Zero(size, 3);
    Eigen::MatrixXf Q  = Eigen::MatrixXf::Zero(size, 3);
    Eigen::MatrixXf NP = Eigen::MatrixXf::Zero(size, 3);
    Eigen::MatrixXf NQ = Eigen::MatrixXf::Zero(size, 3);
    for (int i = 0; i < size; i++) {
        auto &p1 = srcPC.point[ modelIdx[ i ] ];
        auto &n1 = srcPC.normal[ modelIdx[ i ] ];
        auto &p2 = dstPC.point[ sceneIdx[ i ] ];
        auto &n2 = dstPC.normal[ sceneIdx[ i ] ];

        P.row(i)  = p1.transpose();
        Q.row(i)  = p2.transpose();
        NP.row(i) = n1.transpose();
        NQ.row(i) = n2.transpose();
    }

    Eigen::Matrix3f    r;
    Eigen::RowVector3f t;
    symmetric_rigid_matching(P, Q, NP, NQ, r, t);

    Eigen::Isometry3f rt;
    rt.linear()      = r;
    rt.translation() = t.transpose();
    return rt.matrix();
}

std::pair<std::vector<std::size_t>, std::vector<std::size_t>>
    findCorresponds(const PointCloud &srcPC, const KDTree &kdtree, float rejectDist) {
    std::vector<int>   indicies;
    std::vector<float> distances;
    findClosestPoint(kdtree, srcPC, indicies, distances);

    // limit distance
    auto                                          rejectDistSquare = rejectDist * rejectDist;
    std::vector<std::pair<int, std::vector<int>>> map;
    for (int i = 0; i < distances.size(); i++) {
        auto &distance = distances[ i ];
        if (distance > rejectDistSquare)
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
    auto                                                          size = map.size();
    std::pair<std::vector<std::size_t>, std::vector<std::size_t>> result;
    result.first.resize(size);
    result.second.resize(size);
#pragma omp parallel for
    for (int i = 0; i < size; i++) {
        auto &node         = map[ i ];
        result.second[ i ] = node.first;
        result.first[ i ] =
            *std::min_element(node.second.begin(), node.second.end(),
                              [ & ](int a, int b) { return distances[ a ] < distances[ b ]; });
    }

    return result;
}

struct IterResult {
    Eigen::Matrix4f pose       = Eigen::Matrix4f::Identity();
    std::size_t     validPairs = 0;
    float           mse        = std::numeric_limits<float>::max();
};

IterResult iteration(const PointCloud &srcPC, const PointCloud &dstPC, const KDTree &kdtree,
                     float rejectDist) {
    auto modelScenePair = findCorresponds(srcPC, kdtree, rejectDist);
    auto size           = modelScenePair.first.size();
    if (size < 6)
        return IterResult{Eigen::Matrix4f::Identity(), size};

    auto p   = minimizePointToPlaneMetric(srcPC, dstPC, modelScenePair);
    auto pct = transformPointCloud(extraIndices(srcPC, modelScenePair.first), p, true);

    std::vector<int>   indices2;
    std::vector<float> distances2;
    findClosestPoint(kdtree, pct, indices2, distances2);

    float mse = 0;
    for (auto &dist : distances2)
        mse += sqrtf(dist);
    mse /= (float)distances2.size();

    return IterResult{p, modelScenePair.first.size(), mse};
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

    if (criteria_.iterations < 1 || criteria_.mseMin < 0 || criteria_.mseMax < criteria_.mseMin ||
        criteria_.tolerance > 1 || criteria_.tolerance < 0 || criteria_.rejectDist < 0)
        throw std::runtime_error("Invalid ConvergenceCriteria at ICP::regist");

    std::vector<ConvergenceResult> results(initPoses.size());

    for (int i = 0; i < initPoses.size(); i++) {
        auto &initPose = initPoses[ i ];

        ConvergenceResult result;
        result.pose = initPose;
        auto srcTmp = initPose.isIdentity() ? src : transformPointCloud(src, initPose, true);

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

            if (stop)
                break;

            srcTmp = transformPointCloud(srcTmp, tmpResult.pose, true);
        }

        results[ i ] = result;
    }

    return results;
}

float inline square(float x) {
    return x * x;
}

float TukeyLossWeight(float residual, float k) {
    const float e = std::abs(residual);
    return square(1.f - square(std::min(1.f, e / k)));
}

void point_to_plane(const Eigen::MatrixXf &X, const Eigen::MatrixXf &Y, const Eigen::MatrixXf &YN,
                    Eigen::Matrix3f &R, Eigen::RowVector3f &t) {
    auto size = X.rows();

    Eigen::VectorXf w = Eigen::VectorXf::Zero(size);
    Eigen::VectorXf d = Eigen::VectorXf::Zero(size);
#pragma omp parallel for
    for (int i = 0; i < size; ++i) {
        d[ i ] = std::abs(YN.row(i).dot(X.row(i) - Y.row(i)));
        w[ i ] = TukeyLossWeight(d[ i ], 2);
    }

    /// Normalize weight vector
    Eigen::VectorXf wNormalized = w / w.sum();

    /// De-mean
    Eigen::RowVector3f xMean;
    for (int i = 0; i < 3; ++i)
        xMean(i) = (X.col(i).transpose() * wNormalized).sum();
    Eigen::MatrixXf cX = X.rowwise() - xMean;

    /// Prepare LHS and RHS
    Eigen::MatrixXf LHS = Eigen::MatrixXf::Zero(6, 6);
    Eigen::VectorXf RHS = Eigen::VectorXf::Zero(6, 1);

    for (int i = 0; i < size; i++) {
        Eigen::Vector3f x = cX.row(i);
        Eigen::Vector3f n = YN.row(i);

        Eigen::VectorXf J_r   = Eigen::VectorXf::Zero(6);
        J_r.block<3, 1>(0, 0) = x.cross(n);
        J_r.block<3, 1>(3, 0) = n;

        LHS += J_r * w[ i ] * J_r.transpose();
        RHS += J_r * w[ i ] * -d[ i ];
    }
    /// Compute transformation
    RHS = LHS.ldlt().solve(RHS);

    R = Eigen::AngleAxisf(RHS(0), Eigen::Vector3f::UnitX()) *
        Eigen::AngleAxisf(RHS(1), Eigen::Vector3f::UnitY()) *
        Eigen::AngleAxisf(RHS(2), Eigen::Vector3f::UnitZ());

    t = (xMean.transpose() - R * xMean.transpose()).transpose() + RHS.tail<3>().transpose();
}

} // namespace ppf