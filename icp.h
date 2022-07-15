#pragma once

#include <privateType.h>
#include <type.h>

namespace ppf {
struct ConvergenceCriteria {
public:
    /**
     * @brief Convergence Criteria
     *
     * @param iterations max iterations for optimization
     * @param rejectDist rejection for corresponding set
     * @param mseMin min mean-squared-error
     * @param mseMax max mean-squared-error indicate whether be converged
     * @param tolerance min converge rate
     */
    ConvergenceCriteria(int iterations, float rejectDist, float mseMin, float mseMax,
                        float tolerance = 0.05f);
    int   iterations;
    float rejectDist;
    float mseMin;
    float mseMax;
    float tolerance;
};

enum class ConvergenceType {
    ITER,          // reach max iterations
    MSE,           // reach min mean-squared-error
    CONVERGE_RATE, // reach min converge rate
    NO_CORRESPONDS // no  corresponding set
};

struct ConvergenceResult {
    ConvergenceResult();

    Eigen::Matrix4f pose; // last pose refined
    ConvergenceType type; // which cause converge

    float mse;          // last mse
    float convergeRate; // last converge rate
    int   iterations;   // last iteration
    bool  converged;    // whether be converged
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