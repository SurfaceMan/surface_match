#include <helper.h>
#include <icp.h>
#include <kdtree.h>
#include <ppf.h>
#include <privateType.h>
#include <privateUtil.h>
#include <serialize.h>

#include <Eigen/Geometry>
#include <fstream>
#include <map>
#include <numeric>
#include <utility>

#include <filePLY.h>

namespace ppf {

const int VERSION    = 100;
const int MAGIC      = 0x7F27F;
const int MaxThreads = 8;

Detector::Detector()
    : impl_(nullptr) {
#ifdef __OMP_H
    auto numThreads = omp_get_max_threads();
    if (numThreads > MaxThreads)
        omp_set_num_threads(MaxThreads);
#endif
}

Detector::~Detector() {
    if (impl_) {
        delete impl_;
        impl_ = nullptr;
    }
}

void merge(gtl::flat_hash_map<uint32_t, Feature> &out, gtl::flat_hash_map<uint32_t, Feature> &in) {
    for (auto &itm : in) {
        auto &found = out[ itm.first ];
        found.alphaAngle.insert(found.alphaAngle.end(), itm.second.alphaAngle.begin(),
                                itm.second.alphaAngle.end());
        found.refInd.insert(found.refInd.end(), itm.second.refInd.begin(), itm.second.refInd.end());
    }
}

void Detector::trainModel(PointCloud_t model_, float samplingDistanceRel, TrainParam param) {
    if (nullptr == model_)
        throw std::invalid_argument("Invalid Input: model null pointer in trainModel");

    auto model = *model_;
    //[1] check input date
    if (samplingDistanceRel > 1 || samplingDistanceRel < 0)
        throw std::range_error("Invalid Input: samplingDistanceRel range mismatch in trainModel");

    if (model.point.empty())
        throw std::runtime_error("Invalid Input: empty model in trainModel");

    // remove nan
    Timer t0("model remove nan");
    auto  validIndices = removeNan(model, model.hasNormal());
    t0.release();

    if (model.box.diameter() == 0)
        model.box = computeBoundingBox(model, validIndices);

    float modelDiameter = model.box.diameter();

    float sampleStep   = modelDiameter * samplingDistanceRel;
    float reSampleStep = modelDiameter * param.poseRefRelSamplingDistance;
    float distanceStep = modelDiameter * param.featDistanceStepRel;
    float angleStep    = M_2PI / (float)param.featAngleResolution;
    bool  hasNormal    = model.hasNormal();
    int   maxIdx       = param.featAngleResolution - 1;

    // mesh
    if (!model.face.empty()) {
        Timer t("sample mesh");
        model = sampleMesh(model, reSampleStep / 4.f);
        t.release();
        hasNormal = model.hasNormal();
        validIndices.resize(model.size());
        std::iota(validIndices.begin(), validIndices.end(), 0);
    }

    delete impl_;
    impl_                      = new IMPL;
    impl_->samplingDistanceRel = samplingDistanceRel;
    impl_->param               = param;
    impl_->angleTable          = computeAngleTable(param.featAngleResolution);

    Timer  t("model kdtree");
    KDTree kdtree(model.point, 10, model.box, validIndices);
    t.release();
    Timer t1("model sample1");
    auto  indices1 = samplePointCloud(kdtree, sampleStep);
    t1.release();
    Timer t2("model sample2");
    auto  indices2 = samplePointCloud(kdtree, reSampleStep);
    t2.release();

    std::cout << "model point size:" << model.size() << "\n"
              << "model sample step:" << sampleStep << "\n"
              << "model sampled point size:" << indices1.size() << "\n"
              << "model resampled step:" << reSampleStep << "\n"
              << "model resampled point size:" << indices2.size() << std::endl;

    if (!hasNormal) {
        Timer ten("model compute normal");
        estimateNormal(model, indices1, kdtree, param.knnNormal, param.smoothNormal);
        estimateNormal(model, indices2, kdtree, param.knnNormal, param.smoothNormal);
    } else {
        Timer tnn("model normalize normal");
        normalizeNormal(model);
    }

    impl_->sampledModel   = extraIndices(model, indices1);
    impl_->reSampledModel = extraIndices(model, indices2);
    auto &sampledModel    = impl_->sampledModel;

    Timer t3("model ppf");
    //[2] create hash table
    auto size = sampledModel.size();

    // TODO compare difference between reduction and lock
    gtl::flat_hash_map<uint32_t, Feature> hashTable;
#pragma omp declare reduction( \
        mapCombine : gtl::flat_hash_map<uint32_t, Feature> : merge(omp_out, omp_in))
#pragma omp parallel for reduction(mapCombine : hashTable) \
    shared(size, sampledModel, angleStep, distanceStep, maxIdx) default(none)
    for (int i = 0; i < size; i++) {
        auto p1 = sampledModel.point[ i ];
        auto n1 = sampledModel.normal[ i ];

        if (n1.hasNaN())
            continue;
        auto rt = transformRT(p1, n1);

        auto ppf   = computePPF(p1, n1, sampledModel.point.x, sampledModel.point.y,
                                sampledModel.point.z, sampledModel.normal.x, sampledModel.normal.y,
                                sampledModel.normal.z, angleStep, distanceStep);
        auto alpha = computeAlpha(rt, sampledModel.point.x, sampledModel.point.y,
                                  sampledModel.point.z, maxIdx);
        for (int j = 0; j < size; j++) {
            if (i == j)
                continue;
            hashTable[ ppf[ j ] ].push_back(i, alpha[ j ]);
        }
    }
    t3.release();

    impl_->hashTable = std::move(hashTable);
}

void Detector::matchScene(ppf::PointCloud *scene_, std::vector<float> &poses,
                          std::vector<float> &scores, float samplingDistanceRel,
                          float keyPointFraction, float minScore, MatchParam param,
                          MatchResult *matchResult) {
    auto scene = *scene_;
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

    // remove nan
    Timer t0("scene remove nan");
    auto  validIndices = removeNan(scene, scene.hasNormal());
    t0.release();

    if (scene.box.diameter() == 0)
        scene.box = computeBoundingBox(scene, validIndices);

    std::cout << "scene box:" << scene.box.min().transpose() << "<--->"
              << scene.box.max().transpose() << std::endl;

    //[2] prepare data
    //[2.1] data from IMPL
    float modelDiameter   = impl_->sampledModel.box.diameter();
    float angleStep       = M_2PI / (float)impl_->param.featAngleResolution;
    float distanceStep    = modelDiameter * impl_->param.featDistanceStepRel;
    int   angleNum        = impl_->param.featAngleResolution;
    auto  refNum          = impl_->sampledModel.point.size();
    auto &hashTable       = impl_->hashTable;
    auto &modelSampled    = impl_->sampledModel;
    auto &angleTable      = impl_->angleTable;
    int   maxAngleIndex   = angleNum - 1;
    float squaredDiameter = modelDiameter * modelDiameter;
    float reSampleStep    = modelDiameter * impl_->param.poseRefRelSamplingDistance;
    bool  hasNormal       = scene.hasNormal();

    //[2.2] data from keyPointFraction/samplingDistanceRel
    Timer  t("scene kdtree");
    KDTree sceneKdtree(scene.point, 10, scene.box, validIndices);
    t.release();
    Timer   t1("scene sample1");
    float   sampleStep = modelDiameter * samplingDistanceRel;
    VectorI indicesOfSampleScene;
    auto    sampledIndices = samplePointCloud(sceneKdtree, sampleStep, &indicesOfSampleScene);
    t1.release();
    if (!hasNormal) {
        Timer ten("scene compute normal");
        estimateNormal(scene, sampledIndices, sceneKdtree, param.knnNormal, param.smoothNormal,
                       param.invertNormal);
    } else {
        Timer tnn("scene normalize normal");
        normalizeNormal(scene);
    }
    Timer t3("scene sample2");
    sceneKdtree.reduce(indicesOfSampleScene);
    float keySampleStep = sqrtf(1.f / keyPointFraction) * sampleStep;
    auto  keypoint      = samplePointCloud(sceneKdtree, keySampleStep);
    t3.release();
    std::cout << "scene sample step:" << sampleStep << "\n"
              << "scene sampled point size:" << sampledIndices.size() << "\n"
              << "scene keypoint sample step:" << keySampleStep << "\n"
              << "scene keypoint point size:" << keypoint.size() << std::endl;

    //[2.3] data from param
    float voteThreshold  = refNum * minScore;
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

    if (nullptr != matchResult) {
        if (nullptr == matchResult->keyPoint)
            matchResult->keyPoint = new PointCloud;
        if (nullptr == matchResult->sampledScene)
            matchResult->sampledScene = new PointCloud;
        *matchResult->keyPoint     = extraIndices(scene, keypoint);
        *matchResult->sampledScene = extraIndices(scene, sampledIndices);
    }

    Timer             t2("scene ppf");
    std::vector<Pose> poseList;
    auto              end            = hashTable.end();
    auto              accElementSize = angleNum + 1;
    auto              accSize        = refNum * accElementSize;
    VectorI           accumulator(accSize);
    VectorI           idxAngle(1024);

#pragma omp declare reduction(vecCombine : std::vector<Pose> : omp_out.insert( \
        omp_out.end(), omp_in.begin(), omp_in.end()))
#pragma omp parallel for firstprivate(accumulator, idxAngle) reduction(vecCombine : poseList)     \
    shared(keypoint, scene, sceneKdtree, squaredDiameter, voteThreshold, angleStep, distanceStep, \
               accSize, hashTable, end, M_2PI, maxAngleIndex, refNum, angleNum, modelSampled,     \
               accElementSize, maxAngleIndex, angleTable) default(none)
    for (int count = 0; count < keypoint.size(); count++) {
        auto pointIndex = keypoint[ count ];
        auto p1         = scene.point[ pointIndex ];
        auto n1         = scene.normal[ pointIndex ];

        if (n1.hasNaN())
            continue;

        //[3] vote
        std::vector<std::pair<uint32_t, float>> indices;
        auto searched = sceneKdtree.index->radiusSearch(&p1[ 0 ], squaredDiameter, indices,
                                                        nanoflann::SearchParams(32, 0, false));
        if (searched < voteThreshold)
            continue;

        auto    rows = searched - 1;
        VectorF px(rows);
        VectorF py(rows);
        VectorF pz(rows);
        VectorF nx(rows);
        VectorF ny(rows);
        VectorF nz(rows);
        int     neighborCount = 0;
        for (auto &[ idx, dist ] : indices) {
            if (pointIndex == idx)
                continue;

            auto &p             = scene.point;
            auto &n             = scene.normal;
            px[ neighborCount ] = p.x[ idx ];
            py[ neighborCount ] = p.y[ idx ];
            pz[ neighborCount ] = p.z[ idx ];
            nx[ neighborCount ] = n.x[ idx ];
            ny[ neighborCount ] = n.y[ idx ];
            nz[ neighborCount ] = n.z[ idx ];
            neighborCount++;
        }

        auto ppf   = computePPF(p1, n1, px, py, pz, nx, ny, nz, angleStep, distanceStep);
        auto rt    = transformRT(p1, n1);
        auto alpha = computeAlpha(rt, px, py, pz, maxAngleIndex);
        memset(accumulator.data(), 0, accSize * sizeof(int));

        for (std::size_t j = 0; j < ppf.size(); j++) {
            auto alphaScene = alpha[ j ];
            auto hash       = ppf[ j ];
            auto iter       = hashTable.find(hash);
            if (iter == end)
                continue;

            computeVote(accumulator, iter->second.refInd, iter->second.alphaAngle, idxAngle,
                        alphaScene, angleNum, accElementSize, angleTable);
        }

        // [4]nms
        Pose target(0);
        if (!nms(target, accumulator, voteThreshold, refNum, angleNum, accElementSize,
                 maxAngleIndex, modelSampled, rt))
            continue;
        poseList.emplace_back(target);
    }
    t2.release();

    //[5] cluster
    auto clusters = clusterPose(poseList, 0.1f * modelDiameter, angleStep);
    auto avgPoses = avgClusters(clusters);
    auto sorted   = sortPoses(avgPoses);
    auto center   = impl_->sampledModel.box.center();
    auto cluster2 = clusterPose2(sorted, center, maxOverlapDist);

    std::cout << "after cluster has items: " << cluster2.size() << std::endl;

    //[6] icp
    ICP sparseIcp(ConvergenceCriteria(5, poseRefDistThreshold, sampleStep * 0.5f, sampleStep));
    ICP denseIcp(ConvergenceCriteria(param.poseRefNumSteps, poseRefDistThreshold,
                                     reSampleStep * 0.5f, sampleStep));

    VectorI indicesOfSampleScene2;
    if (param.densePoseRefinement) {
        Timer t("icp prepare");
        sceneKdtree.restore();
        auto indices = samplePointCloud(sceneKdtree, reSampleStep, &indicesOfSampleScene2);
        if (!hasNormal)
            estimateNormal(scene, indices, sceneKdtree, param.knnNormal, false, param.invertNormal);
    }

    Timer t4("icp");
    using Target = std::pair<float, Eigen::Matrix4f>;
    std::vector<Target> result; //[score, pose]
    for (auto &p : cluster2) {
        if (p.numVotes < voteThreshold)
            continue;

        float           score;
        Eigen::Matrix4f pose;
        if (!icp(p, score, pose, param, sparseIcp, denseIcp, sceneKdtree, modelSampled,
                 impl_->reSampledModel, scene, indicesOfSampleScene, indicesOfSampleScene2,
                 minScore, poseRefScoringDist))
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
        // poses[ i ]   = target.second;
    }

    std::cout << "after icp has items: " << poses.size() << std::endl;
}

void Detector::save(const std::string &filename) const {
    std::ofstream of(filename, std::ios::out | std::ios::binary);
    if (!of.is_open())
        throw std::runtime_error("failed to open file:" + filename);
    if (!impl_)
        throw std::runtime_error("No trained model in save");

    serialize(&of, MAGIC);
    serialize(&of, VERSION);
    serialize(&of, impl_->samplingDistanceRel);
    serialize(&of, impl_->param);
    serialize(&of, impl_->sampledModel);
    serialize(&of, impl_->reSampledModel);
    serialize(&of, impl_->hashTable);
    serialize(&of, impl_->angleTable);
    of.close();
}

void Detector::load(const std::string &filename) {
    std::ifstream ifs(filename, std::ios::binary);
    if (!ifs.is_open())
        throw std::runtime_error("failed to open file:" + filename);
    if (impl_)
        delete impl_;
    impl_ = new IMPL;

    int magic;
    deserialize(&ifs, magic);
    if (MAGIC != magic)
        throw std::runtime_error("unsupported file format:" + filename);
    int version;
    deserialize(&ifs, version);
    deserialize(&ifs, impl_->samplingDistanceRel);
    deserialize(&ifs, impl_->param);
    deserialize(&ifs, impl_->sampledModel);
    deserialize(&ifs, impl_->reSampledModel);
    deserialize(&ifs, impl_->hashTable);
    deserialize(&ifs, impl_->angleTable);
    ifs.close();
}

} // namespace ppf
