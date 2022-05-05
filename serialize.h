/**
 * @file serialize.h
 * @author your name (you@domain.com)
 * @brief
 * @version 0.1
 * @date 2022-04-27
 *
 * @copyright Copyright (c) 2022
 *
 */

#pragma once

#include <iostream>
#include <private.h>
#include <type.h>

namespace ppf {

void deserialize(std::istream *is, bool &val);
void serialize(std::ostream *os, const bool &val);

void deserialize(std::istream *is, float &val);
void serialize(std::ostream *os, const float &val);

void deserialize(std::istream *is, int &val);
void serialize(std::ostream *os, const int &val);

void deserialize(std::istream *is, uint32_t &val);
void serialize(std::ostream *os, const uint32_t &val);

void deserialize(std::istream *is, Eigen::Vector3f &val);
void serialize(std::ostream *os, const Eigen::Vector3f &val);

void deserialize(std::istream *is, BoundingBox &val);
void serialize(std::ostream *os, const BoundingBox &val);

void deserialize(std::istream *is, std::vector<Eigen::Vector3f> &val);
void serialize(std::ostream *os, const std::vector<Eigen::Vector3f> &val);

void deserialize(std::istream *is, PointCloud &val);
void serialize(std::ostream *os, const PointCloud &val);

void deserialize(std::istream *is, Feature &val);
void serialize(std::ostream *os, const Feature &val);

void deserialize(std::istream *is, std::vector<Feature> &val);
void serialize(std::ostream *os, const std::vector<Feature> &val);

void deserialize(std::istream *is, std::pair<uint32_t, std::vector<Feature>> &val);
void serialize(std::ostream *os, const std::pair<uint32_t, std::vector<Feature>> &val);

void deserialize(std::istream *is, std::unordered_map<uint32_t, std::vector<Feature>> &val);
void serialize(std::ostream *os, const std::unordered_map<uint32_t, std::vector<Feature>> &val);

void deserialize(std::istream *is, TrainParam &val);
void serialize(std::ostream *os, const TrainParam &val);

} // namespace ppf