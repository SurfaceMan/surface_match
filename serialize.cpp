#include <serialize.h>

namespace ppf {

void deserialize(std::istream *is, bool &val) {
    is->read(reinterpret_cast<char *>(&val), sizeof(val));
}

void serialize(std::ostream *os, const bool &val) {
    os->write(reinterpret_cast<const char *>(&val), sizeof(val));
}

void deserialize(std::istream *is, float &val) {
    is->read(reinterpret_cast<char *>(&val), sizeof(val));
}

void serialize(std::ostream *os, const float &val) {
    os->write(reinterpret_cast<const char *>(&val), sizeof(val));
}

void deserialize(std::istream *is, int &val) {
    is->read(reinterpret_cast<char *>(&val), sizeof(val));
}

void serialize(std::ostream *os, const int &val) {
    os->write(reinterpret_cast<const char *>(&val), sizeof(val));
}

void deserialize(std::istream *is, uint32_t &val) {
    is->read(reinterpret_cast<char *>(&val), sizeof(val));
}

void serialize(std::ostream *os, const uint32_t &val) {
    os->write(reinterpret_cast<const char *>(&val), sizeof(val));
}

void deserialize(std::istream *is, Eigen::Vector3f &val) {
    deserialize(is, val.x());
    deserialize(is, val.y());
    deserialize(is, val.z());
}

void serialize(std::ostream *os, const Eigen::Vector3f &val) {
    serialize(os, val.x());
    serialize(os, val.y());
    serialize(os, val.z());
}

void deserialize(std::istream *is, BoundingBox &val) {
    deserialize(is, val.min);
    deserialize(is, val.max);
}

void serialize(std::ostream *os, const BoundingBox &val) {
    serialize(os, val.min);
    serialize(os, val.max);
}

void deserialize(std::istream *is, std::vector<Eigen::Vector3f> &val) {
    uint32_t size;
    deserialize(is, size);
    val.resize(size);
    for (auto &itm : val)
        deserialize(is, itm);
}

void serialize(std::ostream *os, const std::vector<Eigen::Vector3f> &val) {
    uint32_t size = val.size();
    serialize(os, size);
    for (auto &itm : val)
        serialize(os, itm);
}

void deserialize(std::istream *is, PointCloud &val) {
    deserialize(is, val.point);
    deserialize(is, val.normal);
    deserialize(is, val.box);
    deserialize(is, val.viewPoint);
}

void serialize(std::ostream *os, const PointCloud &val) {
    serialize(os, val.point);
    serialize(os, val.normal);
    serialize(os, val.box);
    serialize(os, val.viewPoint);
}

void deserialize(std::istream *is, vectorI &val) {
    uint32_t size;
    deserialize(is, size);
    val.resize(size);
    for (auto &itm : val)
        deserialize(is, itm);
}

void serialize(std::ostream *os, const vectorI &val) {
    uint32_t size = val.size();
    serialize(os, size);
    for (auto &itm : val)
        serialize(os, itm);
}

void deserialize(std::istream *is, vectorF &val) {
    uint32_t size;
    deserialize(is, size);
    val.resize(size);
    for (auto &itm : val)
        deserialize(is, itm);
}

void serialize(std::ostream *os, const vectorF &val) {
    uint32_t size = val.size();
    serialize(os, size);
    for (auto &itm : val)
        serialize(os, itm);
}

void deserialize(std::istream *is, Feature &val) {
    deserialize(is, val.refInd);
    deserialize(is, val.alphaAngle);
}

void serialize(std::ostream *os, const Feature &val) {
    serialize(os, val.refInd);
    serialize(os, val.alphaAngle);
}

void deserialize(std::istream *is, std::vector<Feature> &val) {
    uint32_t size;
    deserialize(is, size);
    val.resize(size);
    for (auto &itm : val)
        deserialize(is, itm);
}

void serialize(std::ostream *os, const std::vector<Feature> &val) {
    uint32_t size = val.size();
    serialize(os, size);
    for (auto &itm : val)
        serialize(os, itm);
}

void deserialize(std::istream *is, std::pair<uint32_t, Feature> &val) {
    deserialize(is, val.first);
    deserialize(is, val.second);
}

void serialize(std::ostream *os, const std::pair<uint32_t, Feature> &val) {
    serialize(os, val.first);
    serialize(os, val.second);
}

void deserialize(std::istream *is, gtl::flat_hash_map<uint32_t, Feature> &val) {
    val.clear();
    uint32_t size;
    deserialize(is, size);
    for (uint32_t i = 0; i < size; i++) {
        std::pair<uint32_t, Feature> itm;
        deserialize(is, itm);
        val.emplace(std::move(itm));
    }
}

void serialize(std::ostream *os, const gtl::flat_hash_map<uint32_t, Feature> &val) {
    uint32_t size = val.size();
    serialize(os, size);
    for (auto &itm : val) {
        serialize(os, itm);
    }
}

void deserialize(std::istream *is, TrainParam &val) {
    deserialize(is, val.featDistanceStepRel);
    deserialize(is, val.featAngleResolution);
    deserialize(is, val.poseRefRelSamplingDistance);
    deserialize(is, val.knnNormal);
    deserialize(is, val.smoothNormal);
}

void serialize(std::ostream *os, const TrainParam &val) {
    serialize(os, val.featDistanceStepRel);
    serialize(os, val.featAngleResolution);
    serialize(os, val.poseRefRelSamplingDistance);
    serialize(os, val.knnNormal);
    serialize(os, val.smoothNormal);
}

} // namespace ppf