#include <type.h>

namespace ppf {
BoundingBox::BoundingBox()
    : min(Eigen::Vector3f::Zero())
    , max(Eigen::Vector3f::Zero()) {
}

BoundingBox::BoundingBox(Eigen::Vector3f min_, Eigen::Vector3f max_)
    : min(min_)
    , max(max_) {
}

BoundingBox &BoundingBox::operator=(const BoundingBox &rhs) {
    if (this != &rhs) {
        this->min = rhs.min;
        this->max = rhs.max;
    }

    return *this;
}

Eigen::Vector3f BoundingBox::size() const {
    return max - min;
}

float BoundingBox::diameter() const {
    return size().norm();
}

bool PointCloud::hasNormal() const {
    return (point.size() > 0) && (point.size() == normal.size());
}

BoxGrid::BoxGrid()
    : step(0)
    , xBins(0)
    , yBins(0)
    , zBins(0) {
}

const int BoxGrid::INVALID;

int BoxGrid::grid2Index(const Eigen::Vector3i &index_) const {
    return index[ index_.x() ][ index_.y() ][ index_.z() ];
}

Eigen::Vector3i BoxGrid::index2Grid(int index) const {
    return grid[ index ];
}

Pose::Pose(float votes)
    : numVotes(votes){};

void Pose::updatePose(const Eigen::Matrix4f &newPose) {
    pose = newPose;

    auto rMatrix = pose.rotation();
    r            = rMatrix;
    q            = rMatrix;
}

void Pose::updatePoseT(const Eigen::Vector3f &t) {
    pose.translation() = t;
}

void Pose::updatePoseQuat(const Eigen::Quaternionf &q_) {
    q             = q_;
    r             = q.matrix();
    pose.linear() = q.matrix();
}

} // namespace ppf