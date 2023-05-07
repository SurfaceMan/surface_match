#include <privateType.h>
#include <type.h>

namespace ppf {

BoundingBox::BoundingBox()
    : BoundingBox(Eigen::Vector3f::Zero(), Eigen::Vector3f::Zero()) {
}

BoundingBox::BoundingBox(Eigen::Vector3f min_, Eigen::Vector3f max_)
    : _min(std::move(min_))
    , _max(std::move(max_))
    , _size(_max - _min)
    , _center((_min + _max) / 2)
    , _diameter(_size.norm()) {
}

BoundingBox &BoundingBox::operator=(const BoundingBox &rhs) {
    if (this != &rhs) {
        this->_min      = rhs._min;
        this->_max      = rhs._max;
        this->_size     = rhs._size;
        this->_center   = rhs._center;
        this->_diameter = rhs._diameter;
    }

    return *this;
}

Eigen::Vector3f BoundingBox::min() const {
    return _min;
}

Eigen::Vector3f BoundingBox::max() const {
    return _max;
}

Eigen::Vector3f BoundingBox::size() const {
    return _size;
}

Eigen::Vector3f BoundingBox::center() const {
    return _center;
}

float BoundingBox::diameter() const {
    return _diameter;
}

PointCloud::PointCloud() {
}

PointCloud::~PointCloud() {
}

bool PointCloud::hasNormal() const {
    return !point.empty() && (point.size() == normal.size());
}

std::size_t PointCloud::size() const {
    return point.size();
}

bool PointCloud::empty() const {
    return point.empty();
}

Pose::Pose(float votes)
    : pose(Eigen::Matrix4f::Identity())
    , numVotes(votes) {
}

void Feature::push_back(uint32_t index, uint32_t angle) {
    refInd.push_back(index);
    alphaAngle.push_back(angle);
}

Candidate::Candidate(float vote_, int refId_, int angleId_)
    : vote(vote_)
    , refId(refId_)
    , angleId(angleId_) {
}

void Pose::updatePose(const Eigen::Matrix4f &newPose) {
    pose = newPose;

    Eigen::Matrix3f rMatrix = pose.topLeftCorner(3, 3);
    r                       = rMatrix;
    q                       = rMatrix;
}

void Pose::updatePoseT(const Eigen::Vector3f &t) {
    pose.topRightCorner(3, 1) = t;
}

void Pose::updatePoseR(const Eigen::Quaternionf &quat) {
    q                        = quat;
    r                        = q.matrix();
    pose.topLeftCorner(3, 3) = q.matrix();
}

TrainParam::TrainParam(float featDistanceStepRel_, int featAngleResolution_,
                       float poseRefRelSamplingDistance_, int knnNormal_, bool smoothNormal_)
    : featDistanceStepRel(featDistanceStepRel_)
    , featAngleResolution(featAngleResolution_)
    , poseRefRelSamplingDistance(poseRefRelSamplingDistance_)
    , knnNormal(knnNormal_)
    , smoothNormal(smoothNormal_) {

    if (featDistanceStepRel > 1 || featDistanceStepRel <= 0)
        throw std::range_error("Invalid Input: featDistanceStepRel range mismatch in TrainParam()");

    if (featAngleResolution <= 0)
        throw std::range_error("Invalid Input: featAngleResolution range mismatch in TrainParam()");

    if (poseRefRelSamplingDistance > 1 || poseRefRelSamplingDistance <= 0)
        throw std::range_error(
            "Invalid Input: poseRefRelSamplingDistance range mismatch in TrainParam()");
}

MatchParam::MatchParam(int numMatches_, int knnNormal_, bool smoothNormal_, bool invertNormal_,
                       float maxOverlapDistRel_, float maxOverlapDistAbs_,
                       bool sparsePoseRefinement_, bool densePoseRefinement_, int poseRefNumSteps_,
                       float poseRefDistThresholdRel_, float poseRefDistThresholdAbs_,
                       float poseRefScoringDistRel_, float poseRefScoringDistAbs_)
    : numMatches(numMatches_)
    , knnNormal(knnNormal_)
    , smoothNormal(smoothNormal_)
    , invertNormal(invertNormal_)
    , maxOverlapDistRel(maxOverlapDistRel_)
    , maxOverlapDistAbs(maxOverlapDistAbs_)
    , sparsePoseRefinement(sparsePoseRefinement_)
    , densePoseRefinement(densePoseRefinement_)
    , poseRefNumSteps(poseRefNumSteps_)
    , poseRefDistThresholdRel(poseRefDistThresholdRel_)
    , poseRefDistThresholdAbs(poseRefDistThresholdAbs_)
    , poseRefScoringDistRel(poseRefScoringDistRel_)
    , poseRefScoringDistAbs(poseRefScoringDistAbs_) {

    if (numMatches < 1)
        throw std::range_error("Invalid Input: numMatches range mismatch in MatchParam()");

    if ((maxOverlapDistRel < 0) && (maxOverlapDistAbs < 0))
        throw std::range_error("Invalid Input: maxOverlapDist range mismatch in MatchParam()");

    if (!densePoseRefinement) {
        if (poseRefNumSteps < 1)
            throw std::range_error("Invalid Input: poseRefNumSteps range mismatch in MatchParam()");

        if ((poseRefDistThresholdRel < 0) && (poseRefDistThresholdAbs < 0))
            throw std::range_error(
                "Invalid Input: poseRefDistThreshold range mismatch in MatchParam()");

        if ((poseRefScoringDistRel < 0) && (poseRefScoringDistAbs < 0))
            throw std::range_error(
                "Invalid Input: poseRefScoringDist range mismatch in MatchParam()");
    }
}

} // namespace ppf
