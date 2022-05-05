/**
 * @file private.h
 * @author your name (you@domain.com)
 * @brief
 * @version 0.1
 * @date 2022-04-27
 *
 * @copyright Copyright (c) 2022
 *
 */

#pragma once

#include <ppf.h>

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
} // namespace ppf