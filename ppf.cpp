#include <helper.h>
#include <icp.h>
#include <ppf.h>
#include <util.h>

#include <Eigen/Geometry>
#include <map>
#include <set>
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

struct Candidate {
public:
    Candidate(float vote_, int refId_, int angleId_)
        : vote(vote_)
        , refId(refId_)
        , angleId(angleId_) {
    }

    float vote = 0;
    int   refId;
    int   angleId;
};

struct Detector::IMPL {
public:
    // model
    float      samplingDistanceRel;
    TrainParam param;

    PointCloud sampledModel;
    PointCloud reSampledModel;

    std::unordered_map<uint32_t, std::vector<Feature>> hashTable;
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

    Timer  t("model kdtree");
    KDTree kdtree(model.point, 10, {model.box.min, model.box.max});
    t.release();
    Timer t1("model sample1");
    impl_->sampledModel = extraIndices(model, samplePointCloud(model, sampleStep, &kdtree));
    t1.release();
    Timer t2("model sample2");
    impl_->reSampledModel = extraIndices(model, samplePointCloud(model, reSampleStep, &kdtree));
    t2.release();

    std::cout << "model sample step:" << sampleStep << "\n"
              << "model sampled point size:" << impl_->sampledModel.point.size() << "\n"
              << "model resampled step:" << reSampleStep << "\n"
              << "model resampled point size:" << impl_->reSampledModel.point.size() << std::endl;

    auto &sampledModel = impl_->sampledModel;
    if (sampledModel.normal.empty())
        sampledModel.normal = estimateNormal(sampledModel, model);
    if (impl_->reSampledModel.normal.empty())
        impl_->reSampledModel.normal = estimateNormal(impl_->reSampledModel, model);

    Timer t3("model ppf");
    //[2] create hash table
    auto    size = sampledModel.size();
    vectorF px(size);
    vectorF py(size);
    vectorF pz(size);
    vectorF nx(size);
    vectorF ny(size);
    vectorF nz(size);
    for (int i = 0; i < size; i++) {
        auto &p = sampledModel.point[ i ];
        auto &n = sampledModel.normal[ i ];
        px[ i ] = p.x();
        py[ i ] = p.y();
        pz[ i ] = p.z();
        nx[ i ] = n.x();
        ny[ i ] = n.y();
        nz[ i ] = n.z();
    }

    std::unordered_map<uint32_t, std::vector<Feature>> hashTable;
#pragma omp parallel for
    for (int i = 0; i < size; i++) {
        auto &p1 = sampledModel.point[ i ];
        auto &n1 = sampledModel.normal[ i ];

        auto ppf   = computePPF(p1, n1, px, py, pz, nx, ny, nz, angleStep, distanceStep);
        auto rt    = transformRT(p1, n1);
        auto alpha = computeAlpha(rt, px, py, pz);
        for (int j = 0; j < size; j++) {
            if (i == j)
                continue;
#pragma omp critical
            { hashTable[ ppf[ j ] ].emplace_back(i, alpha[ j ], 1); }
        }
    }
    t3.release();

    impl_->hashTable = std::move(hashTable);
}

void Detector::matchScene(ppf::PointCloud &scene, std::vector<Eigen::Matrix4f> &poses,
                          std::vector<float> &scores, float samplingDistanceRel,
                          float keyPointFraction, float minScore, MatchParam param,
                          MatchResult *matchResult) {
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
    Timer  t("scene kdtree");
    KDTree sceneKdtree(scene.point, 10, {scene.box.min, scene.box.max});
    t.release();
    Timer t1("scene sample1");
    float sampleStep   = modelDiameter * samplingDistanceRel;
    auto  sampledScene = extraIndices(scene, samplePointCloud(scene, sampleStep, &sceneKdtree));
    if (sampledScene.normal.empty())
        sampledScene.normal = estimateNormal(sampledScene, scene);
    t1.release();
    Timer  t3("scene sample2");
    KDTree kdtree(sampledScene.point, 10, {sampledScene.box.min, sampledScene.box.max});
    float  keySampleStep = sqrtf(1.f / keyPointFraction) * sampleStep;
    auto   keypoint      = samplePointCloud(sampledScene, keySampleStep, &kdtree);
    t3.release();
    std::cout << "scene sample step:" << sampleStep << "\n"
              << "scene sampled point size:" << sampledScene.point.size() << "\n"
              << "scene keypoint sample step:" << keySampleStep << "\n"
              << "scene keypoint point size:" << keypoint.size() << std::endl;

    //[2.3] data from param
    float voteThreshold  = refNum * param.voteThresholdFraction;
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

    if (matchResult) {
        matchResult->keyPoint     = extraIndices(sampledScene, keypoint);
        matchResult->sampledScene = sampledScene;
    }

    Timer              t2("scene ppf");
    std::vector<Pose>  poseList;
    std::vector<float> item(angleNum, 0);
#pragma omp parallel for
    for (int count = 0; count < keypoint.size(); count++) {
        auto  pointIndex = keypoint[ count ];
        auto &p1         = sampledScene.point[ pointIndex ];
        auto &n1         = sampledScene.normal[ pointIndex ];

        //[3] vote
        std::vector<std::pair<std::size_t, float>> indices;
        auto searched = kdtree.index->radiusSearch(&p1[ 0 ], squaredDiameter, indices,
                                                   nanoflann::SearchParams());
        if (searched < voteThreshold)
            continue;

        auto    rows = searched - 1;
        vectorF px(rows);
        vectorF py(rows);
        vectorF pz(rows);
        vectorF nx(rows);
        vectorF ny(rows);
        vectorF nz(rows);
        for (std::size_t i = 0; i < rows; i++) {
            pointIndex = indices[ i + 1 ].first;
            auto &p    = sampledScene.point[ pointIndex ];
            auto &n    = sampledScene.normal[ pointIndex ];
            px[ i ]    = p.x();
            py[ i ]    = p.y();
            pz[ i ]    = p.z();
            nx[ i ]    = n.x();
            ny[ i ]    = n.y();
            nz[ i ]    = n.z();
        }

        auto ppf   = computePPF(p1, n1, px, py, pz, nx, ny, nz, angleStep, distanceStep);
        auto rt    = transformRT(p1, n1);
        auto alpha = computeAlpha(rt, px, py, pz);
        std::vector<std::pair<int, std::vector<float>>> accumulator(refNum, {0, item});
        for (std::size_t j = 1; j < indices.size(); j++) {
            float alphaScene = alpha[ j - 1 ];
            auto  hash       = ppf[ j - 1 ];
            if (hashTable.find(hash) == hashTable.end())
                continue;

            auto &nodeList = hashTable[ hash ];
            for (auto &feature : nodeList) {
                auto &alphaModel = feature.alphaAngle;
                float alphaAngle = alphaModel - alphaScene;
                if (alphaAngle > (float)M_PI)
                    alphaAngle = alphaAngle - M_2PI;
                else if (alphaAngle < (float)(-M_PI))
                    alphaAngle = alphaAngle + M_2PI;

                int   angleIndex = round(maxAngleIndex * (alphaAngle + (float)M_PI) / M_2PI);
                auto &iter       = accumulator[ feature.refInd ];
                iter.first++;
                iter.second[ angleIndex ] += feature.voteValue;
            }
        }

        // [4]nms
        auto cmp = [](const Candidate &a, const Candidate &b) { return a.vote > b.vote; };
        std::multiset<Candidate, decltype(cmp)> maxVal(cmp);

        auto      thre       = voteThreshold / 2.0f;
        const int countLimit = 3;
        for (int i = 0; i < accumulator.size(); i++) {
            if (accumulator[ i ].first < thre)
                continue;

            auto &angles = accumulator[ i ].second;
            auto  iter   = std::max_element(angles.begin(), angles.end());
            int   j      = iter - angles.begin();
            auto  vote   = *iter;
            if (vote < thre)
                continue;

            vote += (j == 0) ? angles[ maxAngleIndex ] : angles[ j - 1 ];
            vote += (j == maxAngleIndex) ? angles[ 0 ] : angles[ j + 1 ];

            maxVal.emplace(vote, i, j);
            if (maxVal.size() > countLimit)
                maxVal.erase(--maxVal.end());
        }
        if (maxVal.empty())
            continue;

        auto iT = rt.inverse();
        thre    = maxVal.begin()->vote * 0.95;
        for (auto &val : maxVal) {
            if (val.vote < thre)
                continue;

            auto &pMax = modelSampled.point[ val.refId ];
            auto &nMax = modelSampled.normal[ val.refId ];

            float           alphaAngle = M_2PI * val.angleId / maxAngleIndex - M_PI;
            Eigen::Matrix4f TPose      = iT * (XRotMat(alphaAngle) * transformRT(pMax, nMax));
            Pose            pose(val.vote);
            pose.updatePose(TPose);

#pragma omp critical
            { poseList.push_back(pose); }
        }
    }
    t2.release();

    //[5] cluster
    auto clusters = clusterPose(poseList, 0.1f * modelDiameter, angleStep);
    auto avgPoses = avgClusters(clusters);
    auto sorted   = sortPoses(avgPoses);
    auto center   = impl_->sampledModel.box.center();
    auto cluster2 = clusterPose2(sorted, center, maxOverlapDist);

    //[6] icp
    ICP sparseIcp(ConvergenceCriteria(5, poseRefDistThreshold, sampleStep, sampleStep * 0.5,
                                      sampleStep * 0.6));
    ICP denseIcp(ConvergenceCriteria(param.poseRefNumSteps, poseRefDistThreshold,
                                     poseRefScoringDist, reSampleStep * 0.5, reSampleStep));

    PointCloud              reSampledScene;
    std::unique_ptr<KDTree> reSampledKdtree;
    if (param.densePoseRefinement) {
        Timer t("icp prepare");
        reSampledScene  = extraIndices(scene, samplePointCloud(scene, reSampleStep, &sceneKdtree));
        reSampledKdtree = std::make_unique<KDTree>(
            reSampledScene.point, 10,
            std::vector<Eigen::Vector3f>{reSampledScene.box.min, reSampledScene.box.max});
    }

    Timer t4("icp");
    using Target = std::pair<float, Eigen::Matrix4f>;
    std::vector<Target> result; //[score, pose]
    for (auto &p : cluster2) {
        auto pose  = p.pose;
        auto score = p.numVotes;

        if (score < voteThreshold)
            continue;

        if (param.sparsePoseRefinement) {
            auto refined = sparseIcp.regist(impl_->sampledModel, sampledScene, kdtree, pose);
            if (!refined.converged)
                continue;

            pose  = refined.pose;
            score = refined.inliner / float(refNum);
            if (score > 1.f)
                score = 1.f;

            std::cout << "sparsePoseRefinement score:" << score << std::endl;
        }

        if (param.sparsePoseRefinement && param.densePoseRefinement) {
            auto refined =
                denseIcp.regist(impl_->reSampledModel, reSampledScene, *reSampledKdtree, pose);
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
    t4.release();

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