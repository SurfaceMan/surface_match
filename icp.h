/**
 * @file icp.h
 * @author y.qiu (y.qiu@pixoel.com)
 * @brief
 * @version 0.1
 * @date 2022-02-17
 *
 * Copyright (c) 2021 Pixoel Technologies Co.ltd.
 *
 */

#pragma once

#include <type.h>

namespace ppf {
struct ConvergenceCriteria {
public:
    /**
     * @brief Convergence Criteria
     *
     * @param iterations_ max iterations for optimization
     * @param inlinerDist_ inliner threshold distance
     * @param mseMin_ min mean-squared-error
     * @param mseMax_ max mean-squared-error indicate whether be converged
     * @param tolerance_ min converge rate
     * @param rejectionScale_ rejection for corresponding set
     */
    ConvergenceCriteria(int iterations_, float inlinerDist_, float mseMin_, float mseMax_,
                        float tolerance_ = 0.05f, float rejectionScale_ = 2.5f);
    int   iterations;
    float inlinerDist;
    float mseMin;
    float mseMax;
    float tolerance;
    float rejectionScale;
};

enum class ConvergenceType {
    ITER,          // reach max iterations
    MSE,           // reach min mean-squared-error
    CONVERGE_RATE, // reach min converge rate
    NO_CORRESPONDS // no  corresponding set
};

struct ConvergenceResult {
    ConvergenceResult();

    bool            converged;    // whether be converged
    ConvergenceType type;         // which cause converge
    float           mse;          // last mse
    float           convergeRate; // last converge rate
    int             iterations;   // last iteration
    Eigen::Matrix4f pose;         // last pose refined
    int             inliner;      // last number of points inliner
};

class ICP {
public:
    explicit ICP(ConvergenceCriteria criteria);
    ~ICP() = default;

    /**
     * @brief register source to target
     *
     * @param src source must has normal
     * @param dst target must has normal
     * @param initPose source initial pose
     * @return ConvergenceResult
     */
    ConvergenceResult regist(const PointCloud &src, const PointCloud &dst,
                             const Eigen::Matrix4f &initPose = Eigen::Matrix4f::Identity()) const;

    /**
     * @brief register source to target
     *
     * @param src source must has normal
     * @param dst target must has normal
     * @param kdtree KDTree for target point
     * @param initPose source initial pose
     * @return ConvergenceResult
     */
    ConvergenceResult regist(const PointCloud &src, const PointCloud &dst, const KDTree &kdtree,
                             const Eigen::Matrix4f &initPose = Eigen::Matrix4f::Identity()) const;

    /**
     * @brief register source to target
     *
     * @param src source
     * @param dst target
     * @param initPose source initial pose
     * @return std::vector<ConvergenceResult> same order as initPose
     */
    std::vector<ConvergenceResult> regist(const PointCloud &src, const PointCloud &dst,
                                          const std::vector<Eigen::Matrix4f> &initPose) const;

    /**
     * @brief register source to target
     *
     * @param src source must has normal
     * @param dst target must has normal
     * @param kdtree KDTree for target point
     * @param initPose source initial pose
     * @return ConvergenceResult
     */
    std::vector<ConvergenceResult> regist(const PointCloud &src, const PointCloud &dst,
                                          const KDTree                       &kdtree,
                                          const std::vector<Eigen::Matrix4f> &initPose) const;

private:
    ConvergenceCriteria criteria_;
};
} // namespace ppf