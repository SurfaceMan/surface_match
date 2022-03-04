/**
 * @file ppf.h
 * @author y.qiu (y.qiu@pixoel.com)
 * @brief
 * @version 0.1
 * @date 2022-02-16
 *
 * Copyright (c) 2021 Pixoel Technologies Co.ltd.
 *
 */

#pragma once

#include <memory>
#include <string>
#include <type.h>

namespace ppf {

class Detector {
public:
    /**
     * @brief Construct a new Detector object
     */
    Detector();
    ~Detector();

    /**
     * @brief train point pair feature
     *
     * @param model model point cloud
     * @param samplingDistanceRel Sampling distance relative to the object's diameter
     * @param param please see TrainParam
     */
    void trainModel(ppf::PointCloud &model, float samplingDistanceRel = 0.04f,
                    TrainParam param = TrainParam());

    /**
     * @brief find matched-model in scene
     *
     * @param scene Scene point cloud
     * @param pose Pose of found model in the scene
     * @param score Score of the found instances of the found model.
     * @param samplingDistanceRel Scene sampling distance relative to the diameter of the model
     * @param keyPointFraction Fraction of sampled scene points used as key points
     * @param minScore Minimum score of the returned poses
     * @param param please see MatchParam
     */
    void matchScene(ppf::PointCloud &scene, std::vector<Eigen::Matrix4f> &pose,
                    std::vector<float> &score, float samplingDistanceRel = 0.04f,
                    float keyPointFraction = 0.2f, float minScore = 0.2f,
                    MatchParam param = MatchParam());

    /**
     * @brief save trained ppf model to file
     *
     * @param filename
     */
    void save(const std::string &filename) const;

    /**
     * @brief load ppf model from file
     *
     * @param filename
     * @return true
     * @return false
     */
    bool load(const std::string &filename);

private:
    struct IMPL;
    std::unique_ptr<IMPL> impl_;
};
} // namespace ppf