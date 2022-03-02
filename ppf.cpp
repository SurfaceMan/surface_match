#include <helper.h>
#include <ppf.h>
#include <util.h>

#include <Eigen/Geometry>
#include <exception>
#include <map>
#include <utility>

#define _USE_MATH_DEFINES
#include <math.h>

const float M_2PI = 2 * M_PI;

namespace ppf {

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
    float featDistanceStepRel;
    int   featAngleResolution;
    float featDistanceStep;
    float featAngleRadians;
    bool  trained;

    ppf::PointCloud                          sampledModel;
    std::map<uint32_t, std::vector<Feature>> hashTable;

    IMPL(float featDistanceStepRel_, int featAngleResolution_)
        : featDistanceStepRel(featDistanceStepRel_)
        , featAngleResolution(featAngleResolution_)
        , featDistanceStep(0)
        , featAngleRadians(360.0f / (float)featAngleResolution / 180.0f * M_PI)
        , trained(false) {
    }
};

Detector::Detector(float featDistanceStepRel, int featAngleResolution)
    : impl_(new IMPL(featDistanceStepRel, featAngleResolution)) {
}

Detector::~Detector() {
}

void Detector::trainModel(ppf::PointCloud &model, float samplingDistanceRel) {
    //[1] check input date
    if (samplingDistanceRel > 1 || samplingDistanceRel < 0)
        throw std::range_error("Invalid Input: samplingDistanceRel range mismatch in trainModel");

    if (model.point.empty())
        throw std::runtime_error("Invalid Input: empty model in trainModel");

    if (model.box.diameter() == 0)
        model.box = computeBoundingBox(model);

    float sampleStep        = model.box.diameter() * samplingDistanceRel;
    auto  sampledModel      = samplePointCloud(model, sampleStep);
    impl_->featDistanceStep = sampledModel.box.diameter() * impl_->featDistanceStepRel;

    std::cout << "model sample step:" << sampleStep << std::endl;

    if (sampledModel.normal.empty())
        sampledModel.normal = estimateNormal(sampledModel, model);

    //[2] create hash table
    std::map<uint32_t, std::vector<Feature>> hashTable;

    float lambda       = 0.98f;
    int   size         = sampledModel.point.size();
    float angleStep    = impl_->featAngleRadians;
    float distanceStep = impl_->featDistanceStep;
    for (int i = 0; i < size; i++) {
        auto &p1 = sampledModel.point[ i ];
        auto &n1 = sampledModel.normal[ i ];
        for (int j = 0; j < size; j++) {
            if (i == j)
                continue;

            auto &p2        = sampledModel.point[ j ];
            auto &n2        = sampledModel.normal[ j ];
            auto  f         = computePPF(p1, p2, n1, n2);
            auto  hash      = hashPPF(f, angleStep, distanceStep);
            auto  alpha     = computeAlpha(p1, p2, n1);
            float dp        = n1.dot(n2);
            float voteValue = 1 - lambda * std::abs(dp); //角度差异越大，投票分数越大

            hashTable[ hash ].emplace_back(i, alpha, voteValue);
        }
    }

    impl_->trained      = true;
    impl_->hashTable    = std::move(hashTable);
    impl_->sampledModel = std::move(sampledModel);
}

void Detector::matchScene(ppf::PointCloud &scene, std::vector<Eigen::Matrix4f> &pose,
                          std::vector<float> &score, float samplingDistanceRel,
                          float keyPointFraction, float minScore, int numMatches) {
    //[1] check input date
    if (!impl_->trained)
        throw std::runtime_error("No trained model in matchScene");

    if (samplingDistanceRel > 1 || samplingDistanceRel < 0)
        throw std::range_error("Invalid Input: samplingDistanceRel range mismatch in matchScene");

    if (keyPointFraction > 1 || keyPointFraction < 0)
        throw std::range_error("Invalid Input: keyPointFraction range mismatch in matchScene");

    if (minScore > 1 || minScore < 0)
        throw std::range_error("Invalid Input: minScore range mismatch in matchScene");

    if (numMatches < 0)
        throw std::range_error("Invalid Input: numMatches range mismatch in matchScene");

    if (scene.point.empty())
        throw std::runtime_error("Invalid Input: empty scene in matchScene");

    if (scene.box.diameter() == 0)
        scene.box = computeBoundingBox(scene);

    //[2] prepare data
    BoxGrid grid;
    float   modelDiameter = impl_->sampledModel.box.diameter();
    float   sampleStep    = modelDiameter * samplingDistanceRel;
    auto    sampledScene  = samplePointCloud(scene, sampleStep, &grid);

    std::cout << "scene sample step:" << sampleStep << std::endl;

    if (sampledScene.normal.empty())
        sampledScene.normal = estimateNormal(sampledScene, scene);

    int   sceneStep     = floor(1 / keyPointFraction);
    int   nStep         = ceil(modelDiameter / grid.step);
    int   size          = sampledScene.point.size();
    int   refNum        = impl_->sampledModel.point.size();
    int   angleNum      = impl_->featAngleResolution;
    int   maxAngleIndex = angleNum - 1;
    float r2            = modelDiameter * modelDiameter;
    float angleStep     = impl_->featAngleRadians;
    float distanceStep  = impl_->featDistanceStep;
    auto &hashTable     = impl_->hashTable;
    auto &modelSampled  = impl_->sampledModel;

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
                    if (count == pointIndex || (p2 - p1).squaredNorm() > r2)
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
                        if (alphaAngle > M_PI)
                            alphaAngle = alphaAngle - 2.0f * M_PI;
                        else if (alphaAngle < (-M_PI))
                            alphaAngle = alphaAngle + 2.0f * M_PI;

                        int angleIndex = round(maxAngleIndex * (alphaAngle + M_PI) / M_2PI);
                        accumulator[ feature.refInd ][ angleIndex ] += feature.voteValue;
                    }
                }
            }
        }

        // [4]nms
        float accuMax = 0;
        for (auto &item1 : accumulator) {
            for (auto &item2 : item1) {
                if (item2 > accuMax)
                    accuMax = item2;
            }
        }

        accuMax = accuMax * 0.95f;
        for (int i = 0; i < accumulator.size(); i++) {
            for (int j = 0; j < angleNum; j++) {
                auto &vote = accumulator[ i ][ j ];
                if (vote <= accuMax)
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
    auto poses    = avgClusters(clusters);
    auto sorted   = sortPoses(poses);

    //[6] icp
    for (int i = 0; i < std::min((int)sorted.size(), numMatches); i++) {
        auto &p = sorted[ i ];
        pose.push_back(p.pose.matrix());
        score.push_back(p.numVotes);
    }

    {
        auto pct = transformPointCloud(impl_->sampledModel, pose[ 0 ]);
        saveText("model.txt", pct);
        saveText("scene.txt", sampledScene);
    }
}

void Detector::save(const std::string &filename) {
}

bool Detector::load(const std::string &filename) {
    return true;
}

} // namespace ppf