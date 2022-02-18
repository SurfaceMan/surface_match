/**
 * @file util.h
 * @author y.qiu (y.qiu@pixoel.com)
 * @brief
 * @version 0.1
 * @date 2022-02-16
 *
 * Copyright (c) 2021 Pixoel Technologies Co.ltd.
 *
 */

#pragma once

#include <chrono>
#include <iostream>
#include <string>

namespace ppf {

class Timer {
public:
    Timer(std::string content)
        : content_(std::move(content))
        , start_(std::chrono::steady_clock::now())
        , released(false) {
    }
    ~Timer() {
        if (!released)
            release();
    }

    void release() {
        released  = true;
        auto end  = std::chrono::steady_clock::now();
        auto cost = std::chrono::duration_cast<std::chrono::milliseconds>(end - start_);
        std::cout << content_ << " cost(ms): " << cost.count() << std::endl;
    }

private:
    std::string                                        content_;
    std::chrono::time_point<std::chrono::steady_clock> start_;
    bool                                               released;
};

} // namespace ppf