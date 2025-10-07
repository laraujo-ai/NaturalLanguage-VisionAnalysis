#pragma once

#include "../../../common/include/interfaces.hpp"
#include <algorithm>
#include <random>

namespace nl_video_analysis {

class UniformFrameSampler : public IFrameSampler {
public:
    UniformFrameSampler() = default;
    void sampleFrames(ClipContainer& clip, int num_frames) override;
};
}