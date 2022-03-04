#include <helper.h>
#include <icp.h>
#include <ppf.h>
#include <util.h>

#include <Eigen/Geometry>
#include <map>
#include <utility>

#define _USE_MATH_DEFINES
#include <math.h>

namespace ppf {

const float M_2PI = 2 * M_PI;

struct Feature {
public:
    int   refInd;
    float alphaAngle;
    float voteValue;

    Feature()
        : refInd(0)
        , alphaAngle(0)
        , voteValue(0) {
    }

    Feature(int refInd_, float alphaAngle_, float voteValue_)
        : refInd(refInd_)
        , alphaAngle(alphaAngle_)
        , voteValue(voteValue_) {
    }
};

struct Detector::IMPL {
public:
    // model
    float      samplingDistanceRel;
    TrainParam param;

    PointCloud sampledModel;
    PointCloud reSampledModel;

    std::map<uint32_t, std::vector<Feature>> hashTable;
};

Detector::Detector()
    : impl_(nullptr) {
}

Detector::~Detector() {
}

void Detector::trainModel(ppf::PointCloud &model, float samplingDistanceRel, TrainParam param) {
    //[1] check input date
    if (samplingDistanceRel > 1 || samplingDistanceRel < 0)
        throw std::range_error("Invalid Input: samplingDistanceRel range mismatch in trainModel");

    if (model.point.empty())
        throw std::runtime_error("Invalid Input: empty model in trainModel");

    if (model.box.diameter() == 0)
        model.box = computeBoundingBox(model);

    float modelDiameter = model.box.diameter();

    float sampleStep   = modelDiameter * samplingDistanceRel;
    float reSampleStep = modelDiameter * param.poseRefRelSamplingDistance;
    float distanceStep = modelDiameter * param.featDistanceStepRel;
    float angleStep    = M_2PI / (float)param.featAngleResolution;

    impl_                      = std::make_unique<IMPL>();
    impl_->samplingDistanceRel = samplingDistanceRel;
    impl_->param               = param;
    impl_->sampledModel        = samplePointCloud(model, sampleStep);
    impl_->reSampledModel      = samplePointCloud(model, reSampleStep);

    std::cout << "model sample step:" << sampleStep << std::endl;

    auto &sampledModel = impl_->sampledModel;
    if (sampledModel.normal.empty())
        sampledModel.normal = estimateNormal(sampledModel, model);

    //[2] create hash table
    std::map<uint32_t, std::vector<Feature>> hashTable;

    // float lambda = 0.98f;
    auto size = sampledModel.point.size();
    for (std::size_t i = 0; i < size; i++) {
        auto &p1 = sampledModel.point[ i ];
        auto &n1 = sampledModel.normal[ i ];
        for (int j = 0; j < size; j++) {
            if (i == j)
                continue;

            auto &p2    = sampledModel.point[ j ];
            auto &n2    = sampledModel.normal[ j ];
            auto  f     = computePPF(p1, p2, n1, n2);
            auto  hash  = hashPPF(f, angleStep, distanceStep);
            auto  alpha = computeAlpha(p1, p2, n1);
            // float dp        = n1.dot(n2);
            float voteValue = 1; // 1 - lambda * std::abs(dp); //角度差异越大，投票分数越大

            hashTable[ hash ].emplace_back(i, alpha, voteValue);
        }
    }

    impl_->hashTable = std::move(hashTable);
}

void Detector::matchScene(ppf::PointCloud &scene, std::vector<Eigen::Matrix4f> &poses,
                          std::vector<float> &scores, float samplingDistanceRel,
                          float keyPointFraction, float minScore, MatchParam param) {
    //[1] check input date
    if (!impl_)
        throw std::runtime_error("No trained model in matchScene");

    if (samplingDistanceRel > 1 || samplingDistanceRel < 0)
        throw std::range_error("Invalid Input: samplingDistanceRel range mismatch in matchScene");

    if (keyPointFraction > 1 || keyPointFraction < 0)
        throw std::range_error("Invalid Input: keyPointFraction range mismatch in matchScene");

    if (minScore > 1 || minScore < 0)
        throw std::range_error("Invalid Input: minScore range mismatch in matchScene");

    if (scene.point.empty())
        throw std::runtime_error("Invalid Input: empty scene in matchScene");

    if (scene.box.diameter() == 0)
        scene.box = computeBoundingBox(scene);

    //[2] prepare data
    //[2.1] data from IMPL
    float modelDiameter   = impl_->sampledModel.box.diameter();
    float angleStep       = M_2PI / (float)impl_->param.featAngleResolution;
    float distanceStep    = modelDiameter * impl_->param.featDistanceStepRel;
    int   angleNum        = impl_->param.featAngleResolution;
    auto  refNum          = impl_->sampledModel.point.size();
    auto &hashTable       = impl_->hashTable;
    auto &modelSampled    = impl_->sampledModel;
    int   maxAngleIndex   = angleNum - 1;
    float squaredDiameter = modelDiameter * modelDiameter;
    float reSampleStep    = modelDiameter * impl_->param.poseRefRelSamplingDistance;

    //[2.2] data from keyPointFraction/samplingDistanceRel
    int sceneStep = floor(1 / keyPointFraction);

    BoxGrid grid;
    float   sampleStep   = modelDiameter * samplingDistanceRel;
    auto    sampledScene = samplePointCloud(scene, sampleStep, &grid);
    int     nStep        = ceil(modelDiameter / grid.step);
    if (sampledScene.normal.empty())
        sampledScene.normal = estimateNormal(sampledScene, scene);

    std::cout << "scene sample step:" << sampleStep << std::endl;

    //[2.3] data from param
    float maxOverlapDist = 0;
    if (param.maxOverlapDistRel > 0)
        maxOverlapDist = modelDiameter * param.maxOverlapDistRel;
    if (param.maxOverlapDistAbs > 0)
        maxOverlapDist = param.maxOverlapDistAbs;

    float poseRefDistThreshold = 0;
    if (param.poseRefDistThresholdRel > 0)
        poseRefDistThreshold = modelDiameter * param.poseRefDistThresholdRel;
    if (param.poseRefDistThresholdAbs > 0)
        poseRefDistThreshold = param.poseRefDistThresholdAbs;

    float poseRefScoringDist = 0;
    if (param.poseRefScoringDistRel > 0)
        poseRefScoringDist = modelDiameter * param.poseRefScoringDistRel;
    if (param.poseRefScoringDistAbs > 0)
        poseRefScoringDist = param.poseRefScoringDistAbs;

    auto              size = sampledScene.point.size();
    std::vector<Pose> poseList;
    for (int count = 0; count < size; count += sceneStep) {
        auto &p1 = sampledScene.point[ count ];
        auto &n1 = sampledScene.normal[ count ];

        //[3] vote
        std::vector<std::vector<float>> accumulator;
        {
            std::vector<float> item(angleNum, 0);
            accumulator.resize(refNum, item);
        }

        Eigen::Matrix3f R;
        Eigen::Vector3f t;
        transformRT(p1, n1, R, t);
        Eigen::Matrix3f iR = R.transpose();
        Eigen::Vector3f it = (-1) * iR * t;
        Eigen::Affine3f iT;
        iT.linear()      = iR;
        iT.translation() = it;

        auto index  = grid.index2Grid(count);
        int  iBegin = std::max(0, index.x() - nStep);
        int  iEnd   = std::min(index.x() + nStep, grid.xBins);
        int  jBegin = std::max(0, index.y() - nStep);
        int  jEnd   = std::min(index.y() + nStep, grid.yBins);
        int  kBegin = std::max(0, index.z() - nStep);
        int  kEnd   = std::min(index.z() + nStep, grid.zBins);
        for (int i = iBegin; i < iEnd; i++) {
            for (int j = jBegin; j < jEnd; j++) {
                for (int k = kBegin; k < kEnd; k++) {
                    int pointIndex = grid.grid2Index({i, j, k});
                    if (pointIndex == BoxGrid::INVALID)
                        continue;

                    auto &p2 = sampledScene.point[ pointIndex ];
                    auto &n2 = sampledScene.normal[ pointIndex ];
                    if (count == pointIndex || (p2 - p1).squaredNorm() > squaredDiameter)
                        continue;

                    auto f    = computePPF(p1, p2, n1, n2);
                    auto hash = hashPPF(f, angleStep, distanceStep);
                    if (hashTable.find(hash) == hashTable.end())
                        continue;

                    Eigen::Vector3f p2t        = R * p2 + t;
                    float           alphaScene = atan2(-p2t(2), p2t(1));
                    if (sin(alphaScene) * p2t(2) > 0)
                        alphaScene = -alphaScene;

                    auto &nodeList = hashTable[ hash ];
                    for (auto &feature : nodeList) {
                        auto  alphaModel = feature.alphaAngle;
                        float alphaAngle = alphaModel - alphaScene;
                        if (alphaAngle > (float)M_PI)
                            alphaAngle = alphaAngle - 2.0f * (float)M_PI;
                        else if (alphaAngle < (float)(-M_PI))
                            alphaAngle = alphaAngle + 2.0f * (float)M_PI;

                        int angleIndex = round(maxAngleIndex * (alphaAngle + (float)M_PI) / M_2PI);
                        accumulator[ feature.refInd ][ angleIndex ] += feature.voteValue;
                    }
                }
            }
        }

        // [4]nms
        float maxVal = 0;
        for (auto &item1 : accumulator) {
            for (auto &item2 : item1) {
                if (item2 > maxVal)
                    maxVal = item2;
            }
        }

        maxVal = maxVal * 0.95f;
        for (int i = 0; i < accumulator.size(); i++) {
            for (int j = 0; j < angleNum; j++) {
                auto &vote = accumulator[ i ][ j ];
                if (vote <= maxVal)
                    continue;

                auto            pMax = modelSampled.point[ i ];
                auto            nMax = modelSampled.normal[ i ];
                Eigen::Matrix3f RMax;
                Eigen::Vector3f tMax;
                transformRT(pMax, nMax, RMax, tMax);
                Eigen::Affine3f TMax;
                TMax.linear()      = RMax;
                TMax.translation() = tMax;

                float           alphaAngle = M_2PI * j / maxAngleIndex - M_PI;
                Eigen::Matrix4f TAlpha     = XRotMat(alphaAngle);
                Eigen::Matrix4f TPose      = iT * (TAlpha * TMax.matrix());
                Pose            pose(vote);
                pose.updatePose(TPose);

                poseList.push_back(pose);
            }
        }
    }

    //[5] cluster
    auto clusters = clusterPose(poseList, 0.1f * modelDiameter, angleStep);
    auto avgPoses = avgClusters(clusters);
    auto sorted   = sortPoses(avgPoses);
    auto center   = impl_->sampledModel.box.center();
    auto cluster2 = clusterPose2(sorted, center, maxOverlapDist);

    std::cout << "sampledModel:" << impl_->sampledModel.point.size() << "\n"
              << "sampledScene:" << sampledScene.point.size() << std::endl;

    //[6] icp
    ICP sparseIcp(ConvergenceCriteria(5, sampleStep, sampleStep * 0.5, sampleStep * 0.6));
    ICP denseIcp(ConvergenceCriteria(param.poseRefNumSteps, reSampleStep, reSampleStep * 0.5,
                                     reSampleStep * 0.6));

    PointCloud reSampledScene;
    if (param.densePoseRefinement) {
        reSampledScene = samplePointCloud(scene, reSampleStep);
    }

    using Target = std::pair<float, Eigen::Matrix4f>;
    std::vector<Target> result; //[score, pose]
    for (auto &p : cluster2) {
        auto pose  = p.pose.matrix();
        auto score = p.numVotes;

        if (param.sparsePoseRefinement) {
            auto refined = sparseIcp.regist(impl_->sampledModel, sampledScene, pose);
            if (!refined.converged)
                continue;

            pose  = refined.pose;
            score = refined.inliner / float(refNum);
            if (score > 1.f)
                score = 1.f;

            std::cout << "sparsePoseRefinement score:" << score << std::endl;
        }

        if (param.sparsePoseRefinement && param.densePoseRefinement) {
            auto refined = denseIcp.regist(impl_->reSampledModel, reSampledScene, pose);
            if (!refined.converged)
                continue;

            pose  = refined.pose;
            score = refined.inliner / float(impl_->reSampledModel.point.size());
            if (score > 1.f)
                score = 1.f;

            std::cout << "densePoseRefinement score:" << score << std::endl;
        }

        if ((param.sparsePoseRefinement || param.densePoseRefinement) && (score < minScore))
            continue;

        result.emplace_back(score, pose);
        if (result.size() >= param.numMatches)
            break;
    }

    std::sort(result.begin(), result.end(),
              [](const Target &a, const Target &b) { return a.first > b.first; });

    scores.resize(result.size());
    poses.resize(result.size());
    for (std::size_t i = 0; i < result.size(); i++) {
        auto &target = result[ i ];
        scores[ i ]  = target.first;
        poses[ i ]   = target.second;
    }
}

void Detector::save(const std::string &filename) const {
}

bool Detector::load(const std::string &filename) {
    return true;
}

} // namespace ppf